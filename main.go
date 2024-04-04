package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/prompts"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores"
	"github.com/tmc/langchaingo/vectorstores/chroma"
)

func loadGutenberg(targetURL string) []string {
	var ret []string
	var buff string
	reject := true

	file, err := os.Open(targetURL)
	if err != nil {
		fmt.Println("Error opening file:", err)
		return nil
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		strippedLine := strings.TrimSpace(line)

		if reject {
			if strings.HasPrefix(strippedLine, "*** START OF THIS PROJECT GUTENBERG EBOOK") {
				reject = false
				continue
			}
		} else {
			if strings.HasPrefix(strippedLine, "*** END OF THIS PROJECT GUTENBERG EBOOK") {
				reject = true
				continue
			}
			if strippedLine != "" {
				if strings.HasPrefix(strippedLine, "=") && strings.HasSuffix(strippedLine, "=") {
					ret = append(ret, buff)
					buff = ""
					buff += strippedLine[1:len(strippedLine)-1] + "\n\n"
				} else {
					buff += strings.ReplaceAll(line, "\r", "")
				}
			}
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Println("Error reading file:", err)
		return nil
	}

	if strings.TrimSpace(buff) != "" {
		ret = append(ret, buff)
	}

	fmt.Printf("Loaded %d documents\n", len(ret))
	return ret
}

func getPromptForLLM(promptTemplate prompts.PromptTemplate, query string, vectorStore vectorstores.VectorStore) string {
	retriever := vectorstores.ToRetriever(vectorStore, 2, vectorstores.WithScoreThreshold(0.50))
	resDocs, err := retriever.GetRelevantDocuments(context.Background(), query)
	if err != nil {
		log.Fatal(err)
	}

	// Transform the list of documents into a list of strings
	resStrings := []string{}
	for _, doc := range resDocs {
		resStrings = append(resStrings, doc.PageContent)
	}

	prompt, err := promptTemplate.Format(map[string]any{
		"context": strings.Join(resStrings, "\n"),
		"query":   query,
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(prompt)
	return prompt
}

func main() {
	promptTpl := prompts.NewPromptTemplate(
		"Human: Context information is below.\n---------------------\n{{.context}}\n---------------------\nGiven the context information and not prior knowledge, answer the query. Please be brief, concise, and complete.\nIf the context information does not contain an answer to the query, respond with \"No information\".Query: {{.query}}\nAssistant: ",
		[]string{"context", "query"},
	)

	ollamaEmbedderLLM, err := ollama.New(ollama.WithModel("znbang/bge:large-en-v1.5-f32"))
	if err != nil {
		log.Fatal(err)
	}
	ollamaEmbedder, err := embeddings.NewEmbedder(ollamaEmbedderLLM)
	if err != nil {
		log.Fatal(err)
	}

	// Create a new Chroma vector store.
	store, errNs := chroma.New(
		chroma.WithNameSpace("linky"),
		chroma.WithChromaURL("http://192.168.1.64:8001"),
		chroma.WithEmbedder(ollamaEmbedder),
		chroma.WithDistanceFunction("cosine"),
	)
	if errNs != nil {
		log.Fatalf("new: %v\n", errNs)
	}

	pages := loadGutenberg("./computers on the farm.txt")
	fmt.Println("Loaded", len(pages), "pages")

	docs := make([]schema.Document, len(pages))
	for i, line := range pages {
		docs[i] = schema.Document{PageContent: line}
	}

	// Add documents to the vector store.
	_, errAd := store.AddDocuments(context.Background(), docs)
	if errAd != nil {
		log.Fatalf("AddDocument: %v\n", errAd)
	}

	ollamaLLM, err := ollama.New(ollama.WithModel("gemma:2b"))
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()

	searchPrompt := getPromptForLLM(promptTpl, "What is the address of AgriData Resources?", store)
	embeddedResponse, err := ollamaLLM.Call(ctx, searchPrompt,
		llms.WithTemperature(0.8),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(embeddedResponse)
}
