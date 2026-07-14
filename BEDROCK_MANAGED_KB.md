# Bedrock Managed Knowledge Base Support

## Changes
- Added `template_managed.yml` CloudFormation template for workshop managed KB setup
- New workshop module demonstrating managed KB creation without vector store provisioning
- Retrieval notebook updated with `managedSearchConfiguration` examples
- AgenticRetrieveStream workshop section added for agentic retrieval pattern
- Original VECTOR workshop modules and templates unchanged

## Design
- MANAGED available via separate template_managed.yml; original VECTOR template unchanged
- Separate CFN template simplifies workshop setup (no OpenSearch/vector store needed)
- Workshop flow: create managed KB → ingest documents → retrieve with managed search
- AgenticRetrieveStream demonstrated as advanced retrieval option

## API Shapes
- KB Creation: `Type: MANAGED` + `ManagedKnowledgeBaseConfiguration.EmbeddingModelType: MANAGED`
- Data Source: `Type: MANAGED_KNOWLEDGE_BASE_CONNECTOR`
- Retrieval: `managedSearchConfiguration` (not `vectorSearchConfiguration`)
- Agentic: `AgenticRetrieveStream` with `foundationModelType: MANAGED`, `rerankingModelType: MANAGED`

## Configuration
| Parameter | Description | Default |
|---|---|---|
| KnowledgeBaseType | MANAGED or VECTOR (template selection) | MANAGED |
| UseAgenticRetrieval | Enable agentic retrieval in examples | true |

## SDK Requirements
- boto3 >= 1.43 for managed search and agentic retrieval
- CloudFormation support for `AWS::Bedrock::KnowledgeBase` Type: MANAGED

## Required IAM Permissions
```json
{
  "Effect": "Allow",
  "Action": [
    "bedrock:Retrieve",
    "bedrock:AgenticRetrieve"
  ],
  "Resource": "arn:aws:bedrock:<region>:<account-id>:knowledge-base/<kb-id>"
}
```
