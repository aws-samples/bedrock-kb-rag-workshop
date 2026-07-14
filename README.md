# Retrieval Augmented Generation using Amazon Bedrock

This repository provides sample code for implementing a question answering application using the Retrieval Augmented Generation (RAG) technique with Amazon Bedrock. A RAG implementation consists of two parts:

1. A data pipeline that ingests that from documents (typically stored in Amazon S3) into a knowledge base i.e. a vector database such as Amazon OpenSearch Service Serverless (AOSS) so that it is available for lookup when a question is received.

1. An application that receives a question from the user, looks up the knowledge base for relevant pieces of information (context) and then creates a prompt that includes the question and the context and provides it to an LLM for generating a response.

The data pipeline represents an undifferentiated heavy lifting and can be implemented using Amazon Bedrock Agents for knowledge Base. We can now connect an S3 bucket to a vector database such as AOSS and have a Bedrock Agent read the objects (html, pdf, text etc.), chunk them, and then convert these chunks into embeddings using Amazon Titan Embeddings model and then store these embeddings in AOSS. All of this without having to build, deploy and manage the data pipeline.

Once the data is available in the Bedrock Knowledge Base then a question answering application can be built using the following architectural pattern.

![KB Agent](img/ML-15729-bedrock-agents-kb.png)

## Installation

Follow the steps listed below to create and run the RAG solution. The [blog_post.md](./blog_post.md) describes this solution in detail.

1. Launch the AWS CloudFormation template included in this repository using one of the buttons from the table below. The CloudFormation template creates the following resources within your AWS account: Amazon OpenSearch Service Serverless (AOSS) Collection, Amazon S3 bucket, IAM roles for Amazon Bedrock Knowledge Base Agent and Notebook and a Amazon SageMaker Notebook with this repository cloned to run the next steps.


   |AWS Region                |     Link        |
   |:------------------------:|:-----------:|
   |us-east-1 (N. Virginia)    | [<img src="./img/ML-15729-cloudformation-launch-stack.png">](https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/new?stackName=rag-w-bedrock-kb&templateURL=https://aws-blogs-artifacts-public.s3.amazonaws.com/artifacts/ML-15729/template.yml) |
   |us-west-2 (Oregon)          | [<img src="./img/ML-15729-cloudformation-launch-stack.png">](https://console.aws.amazon.com/cloudformation/home?region=us-west-2#/stacks/new?stackName=rag-w-bedrock-kb&templateURL=https://aws-blogs-artifacts-public.s3.amazonaws.com/artifacts/ML-15729/template.yml) |

1. Follow instructions in [Build a RAG based question answer solution using Amazon Bedrock Knowledge Base and Amazon OpenSearch Service Serverless](./blog_post.md)

## Managed Knowledge Bases (New)

This workshop was originally built with a **vector knowledge base** using Amazon OpenSearch Serverless. Amazon Bedrock now also supports **Managed Knowledge Bases**, which simplify the setup by eliminating the need for a separate vector store.

With Managed Knowledge Bases:
- Bedrock handles embedding, storage, and retrieval automatically
- No OpenSearch Serverless collection required (no minimum OCU costs)
- Supports agentic retrieval with intelligent query decomposition and managed reranking

A managed knowledge base CloudFormation template is provided at [`template_managed.yml`](./template_managed.yml) as an alternative to the vector-based `template.yml`.

To use managed retrieval in your application code:
```python
import boto3

client = boto3.client('bedrock-agent-runtime')

response = client.retrieve(
    knowledgeBaseId='YOUR_KB_ID',
    retrievalQuery={'text': 'your question'},
    retrievalConfiguration={
        'managedSearchConfiguration': {
            'numberOfResults': 5
        }
    }
)
```

> **SDK requirements:** `boto3 >= 1.43` for managed search and agentic retrieval.

### Reranking Options

Managed KBs use a service-managed reranker by default. You can customize this in the `managedSearchConfiguration`:
- `"rerankingModelType": "MANAGED"` (default) — automatic reranking, no config needed
- `"rerankingModelType": "NONE"` — disable reranking
- `"rerankingModelType": "CUSTOM"` — use your own Bedrock reranking model (e.g., Cohere Rerank v3.5)

### Resources

- [Build a Managed Knowledge Base](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-build-managed.html)
- [Create a Managed Knowledge Base](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-managed-create.html)
- [Query a Knowledge Base (Retrieve API)](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-test-retrieve.html)
- [Connect a Data Source (Web Crawler)](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-managed-ds-webcrawler.html)
- [Agentic Retrieval](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-test-agentic.html)

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](./LICENSE) file.


