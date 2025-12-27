# Deployment Guide

## Table of Contents
1. [Local Deployment](#local-deployment)
2. [Docker Deployment](#docker-deployment)
3. [Model Storage & S3 Setup](#model-storage--s3-setup)
4. [Cloud Deployment](#cloud-deployment)
5. [Production Considerations](#production-considerations)

## Local Deployment

### Prerequisites
- Python 3.8+
- Virtual environment
- 8GB+ RAM (16GB recommended for transformer models)

### Steps

```bash
# 1. Clone repository
git clone https://github.com/irfanfetahovic/sentiment-analysis.git
cd sentiment-analysis

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download trained model (if not training from scratch)
# Place model files in models/distilbert_sentiment/ or models/classical_models

# 5. Run API server
# FastAPI (default):
python app/app_fastapi.py

# Or Flask alternative:
python app/app.py
```

API will be available at `http://localhost:5000`

## Docker Deployment

### Image Size Note

The Docker image is approximately **3.5GB** due to PyTorch and Transformers dependencies. This is typical for ML inference containers. For this reason:
- **CI/CD builds the image but does not push to a registry** (to avoid slow pushes/pulls)
- For local development/demo, build the image locally
- For production, consider using ONNX Runtime to reduce size to ~700MB

### Using a Container Registry (Optional)

If you want to push to a registry, you have several options:

**GitHub Container Registry (GHCR):**
```bash
# Login to GHCR
echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_USERNAME --password-stdin

# Tag and push
docker tag sentiment-analysis:latest ghcr.io/YOUR_USERNAME/sentiment-analysis:latest
docker push ghcr.io/YOUR_USERNAME/sentiment-analysis:latest

# Pull from GHCR
docker pull ghcr.io/YOUR_USERNAME/sentiment-analysis:latest
```

**Docker Hub:**
```bash
# Login to Docker Hub
docker login -u YOUR_USERNAME

# Tag and push
docker tag sentiment-analysis:latest YOUR_USERNAME/sentiment-analysis:latest
docker push YOUR_USERNAME/sentiment-analysis:latest
```

> **Note:** Large images (~3.5GB) will take significant time to push/pull. Consider this for CI/CD pipeline duration.

### Build Optimization

The project includes a `.dockerignore` file that excludes unnecessary files (tests, notebooks, docs, large data files) to:
- Reduce build context size (~100x smaller)
- Speed up builds (30-60 seconds faster)
- Prevent build timeouts

### Build and Run

```bash
# Build Docker image
docker build -t sentiment-analysis:latest .

# Run container ("%cd% for CMD use, ${PWD} for Power Shell)
docker run -d \
  -p 5000:5000 \
  -v $(PWD)/models:/app/models:ro \ 
  --name sentiment-api \
  sentiment-analysis:latest

# Check logs
docker logs -f sentiment-api

# Stop container
docker stop sentiment-api
```

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Model Storage & S3 Setup

### Overview

This project follows professional ML deployment practices by **separating code from models**:
- **Docker image**: Contains application code only (~300MB)
- **Model files**: Stored externally in S3, GitHub Releases, or similar
- **Benefits**: Faster builds, independent versioning, lower costs, deployment flexibility

The Docker image includes an entrypoint script that automatically downloads models on container startup.

### Model Preparation

Package your trained model as a tar.gz archive:

```bash
# From project root
cd models
tar -czf distilbert_sentiment.tar.gz distilbert_sentiment/
tar -czf classical_models.tar.gz classical_models/


# Verify archive contents
tar -tzf distilbert_sentiment.tar.gz
```

The archive should contain:
```
distilbert_sentiment/
├── config.json
├── model.safetensors
├── vocab.txt
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

### Upload to S3

#### Using AWS Console

1. Go to S3 → Create bucket (e.g., `irfan-ml-models`)
2. Upload `distilbert_sentiment.tar.gz` to desired path
3. Note the URI: `s3://my-ml-models/sentiment-analysis/distilbert_sentiment.tar.gz`

#### Using AWS CLI

```bash
# Create bucket (one time)
aws s3 mb s3://irfan-ml-models

# Upload model
aws s3 cp models/distilbert_sentiment.tar.gz \
  s3://irfan-ml-models/sentiment-analysis/distilbert_sentiment.tar.gz

aws s3 cp models/classical_models.tar.gz \
  s3://irfan-ml-models/sentiment-analysis/classical_models.tar.gz

# Verify upload
aws s3 ls s3://my-ml-models/sentiment-analysis/

# Set private access (recommended)
aws s3api put-object-acl \
  --bucket my-ml-models \
  --key sentiment-analysis/distilbert_sentiment.tar.gz \
  --acl private
```

### IAM Permissions

#### Option A: IAM Role (Recommended for AWS deployments)

Create an IAM role with S3 read permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::my-ml-models/*",
        "arn:aws:s3:::my-ml-models"
      ]
    }
  ]
}
```

Attach this role to your ECS task definition or EC2 instance profile.

#### Option B: Access Keys (For non-AWS deployments)

Create an IAM user with the above policy, generate access keys, and use as environment variables.

### Container Deployment with S3 Models

#### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `MODEL_S3_URI` | No* | S3 URI for model (e.g., `s3://bucket/path/model.tar.gz`) |
| `MODEL_URL` | No* | HTTP URL for model (GitHub Releases, etc.) |
| `AWS_ACCESS_KEY_ID` | If using S3 without IAM | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | If using S3 without IAM | AWS secret key |
| `AWS_DEFAULT_REGION` | If using S3 | AWS region (e.g., `us-east-1`) |

*Set either `MODEL_S3_URI` or `MODEL_URL`, not both.

#### Local Testing

```bash
docker run -d \
  -p 5000:5000 \
  -e MODEL_S3_URI=s3://my-ml-models/sentiment-analysis/distilbert_sentiment.tar.gz \
  -e AWS_ACCESS_KEY_ID=YOUR_KEY \
  -e AWS_SECRET_ACCESS_KEY=YOUR_SECRET \
  -e AWS_DEFAULT_REGION=us-east-1 \
  sentiment-analysis:latest
```

#### Using GitHub Releases (Alternative to S3)

If you prefer not to use S3:

```bash
# Create release and upload model
gh release create v1.0.0 models/distilbert_sentiment.tar.gz

# Deploy with MODEL_URL
docker run -d \
  -p 5000:5000 \
  -e MODEL_URL=https://github.com/your-username/sentiment-analysis/releases/download/v1.0.0/distilbert_sentiment.tar.gz \
  sentiment-analysis:latest
```

#### Using Persistent Volumes (Production)

To avoid re-downloading models on every container restart:

```bash
# Create named volume
docker volume create sentiment-models

# Run with volume mount
docker run -d \
  -p 5000:5000 \
  -v sentiment-models:/app/models \
  -e MODEL_S3_URI=s3://my-ml-models/sentiment-analysis/distilbert_sentiment.tar.gz \
  -e AWS_ACCESS_KEY_ID=YOUR_KEY \
  -e AWS_SECRET_ACCESS_KEY=YOUR_SECRET \
  sentiment-analysis:latest
```

Model downloads once, then persists across container restarts.

### How It Works

1. Container starts → Entrypoint script executes
2. Checks if `/app/models/distilbert_sentiment` exists
3. If missing and `MODEL_S3_URI` or `MODEL_URL` is set → Downloads model
4. Extracts model to `/app/models/`
5. Starts gunicorn application server
6. Model remains cached for container lifetime (or in persistent volume)

### Cost Considerations

**S3 Storage:**
- Standard tier: ~$0.023/GB/month
- Example: 500MB model = ~$0.01/month

**Data Transfer:**
- Within AWS (same region): Free
- AWS to internet: $0.09/GB after 100GB/month free tier
- Example: 500MB model × 100 container restarts/month = $4.50

**Optimization:** Use persistent volumes (EFS, EBS) to cache models and minimize S3 downloads.

### Troubleshooting

**Model download fails:**

```bash
# Check container logs
docker logs <container-id>

# Verify S3 access from container
docker exec -it <container-id> aws s3 ls s3://my-ml-models/

# Test manual download
docker exec -it <container-id> bash
aws s3 cp $MODEL_S3_URI /tmp/test.tar.gz
```

**Slow startup / health check fails:**

- Model download adds 5-30 seconds to startup time
- Health check `start-period` is set to 90s to accommodate downloads
- Increase if your model is larger: Edit `HEALTHCHECK --start-period=120s` in Dockerfile

**Container starts but predictions fail:**

```bash
# Check if model was downloaded
docker exec -it <container-id> ls -la /app/models/distilbert_sentiment/

# Check environment variables
docker exec -it <container-id> env | grep MODEL
```

## Cloud Deployment

### AWS Deployment

#### Option 1: AWS Elastic Beanstalk

```bash
# Install EB CLI
pip install awsebcli

# Initialize EB application
eb init -p docker sentiment-analysis

# Create environment and deploy
eb create sentiment-analysis-env
eb deploy

# Open application
eb open
```

#### Option 2: AWS ECS (Fargate)

**Prerequisites**: Create ECR repository first (via AWS Console or CLI):

```bash
aws configure # login into IAM user which contain necessary privilages
# needed: aws configure command and setting up aws cli user
# accound-id is 12-digit number and can be found with aws cli command "aws sts get-caller-identity"
# aws cli user should have also AmazonEC2ContainerRegistryPowerUser and AmazonEC2ContainerRegistryFullAccess policy
aws ecr create-repository --repository-name sentiment-analysis
```

**Build and Push to ECR:**

```bash
# 1. Build Docker image locally
docker build -t sentiment-analysis:latest .

# 2. Authenticate with ECR

aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com

# 3. Tag for ECR
docker tag sentiment-analysis:latest <account-id>.dkr.ecr.<region>.amazonaws.com/sentiment-analysis:latest

# 4. Push to ECR
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/sentiment-analysis:latest
```

<!-- Alternative: If you push to GHCR, you can pull from there instead:
# docker pull ghcr.io/<your-github-username>/sentiment-analysis:latest
# docker tag ghcr.io/<your-github-username>/sentiment-analysis:latest <account-id>.dkr.ecr.<region>.amazonaws.com/sentiment-analysis:latest
# docker push <account-id>.dkr.ecr.<region>.amazonaws.com/sentiment-analysis:latest
-->

**Next Steps**: Create ECS task definition and service

# 1. Create ECS Task definitions

You must register a task definition before creating a service. Below is a professional, secure sample for this project:

> **Security Best Practice:**
> Do NOT include AWS credentials or secrets directly in your task definition or version control. Use IAM roles for tasks (recommended) or reference secrets from AWS Secrets Manager or SSM Parameter Store.

1. Create a file named `ecs-task-definition.json` in your project root with the following content (replace `<account-id>`, `<region>`, and ARNs as needed):

```json
{
  "family": "sentiment-analysis",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::<account-id>:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::<account-id>:role/sentimentAppTaskRole",
  "containerDefinitions": [
    {
      "name": "sentiment-api",
      "image": "<account-id>.dkr.ecr.<region>.amazonaws.com/sentiment-analysis:latest",
      "portMappings": [
        {
          "containerPort": 5000,
          "protocol": "tcp"
        }
      ],
      "essential": true,
      "environment": [
        { "name": "MODEL_S3_URI", "value": "s3://my-ml-models/sentiment-analysis/distilbert_sentiment.tar.gz" },
        { "name": "AWS_DEFAULT_REGION", "value": "<region>" }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/sentiment-analysis",
          "awslogs-region": "<region>",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:5000/health || exit 1"],
        "interval": 30,
        "timeout": 10,
        "retries": 3,
        "startPeriod": 90
      }
    }
  ]
}
```

- `executionRoleArn` and `taskRoleArn` should be IAM roles with permissions to pull from ECR, write logs, and access S3/SSM as needed.


2. Register the task definition:

```bash
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json
```

After successful registration, use the outputted task definition revision (e.g., sentiment-analysis:1) in the next step.

# 2. Create ECS service

## First create a cluster
aws ecs create-cluster `
  --region eu-north-1 `
  --cluster-name sentiment-cluster

## Then choose subnet and security group, and create log group

List of subnets to choose from
aws ec2 describe-subnets --region eu-north-1 --query "Subnets[*].{ID:SubnetId,Name:Tags[?Key=='Name']|[0].Value}" 

List of security groups to choose from
aws ec2 describe-security-groups --region eu-north-1 --query "SecurityGroups[*].{ID:GroupId,Name:GroupName}"


aws logs create-log-group --log-group-name /ecs/sentiment-analysis --region eu-north-1



## Create service
Replace Replace subnet-XXXX and sg-YYYY with the IDs you found in the first two commands in the previous section.

aws ecs create-service `
  --region eu-north-1 `
  --cluster sentiment-cluster `
  --service-name sentiment-service `
  --task-definition sentiment-analysis:1 `# 1 is a revision number
  --desired-count 1 `
  --launch-type FARGATE `
  --network-configuration "awsvpcConfiguration={subnets=[subnet-XXXX],securityGroups=[sg-YYYY],assignPublicIp=ENABLED}"

## Useful commands
aws ecs describe-services --cluster sentiment-cluster --services sentiment-service --region eu-north-1
aws ecs describe-tasks --cluster sentiment-cluster --tasks xxxxxxxxxxxxxxxxxxxx --region eu-north-1
aws ecs delete-service --cluster sentiment-cluster --service sentiment-service --force --region eu-north-1
aws ecs update-service --cluster sentiment-cluster --service sentiment-service --force-new-deployment --region eu-north-1

## Testing 

aws ecs list-tasks --cluster sentiment-cluster --service-name sentiment-service --region eu-north-1


#### Option 3: AWS Lambda + API Gateway


For serverless deployment:
```python
# lambda_function.py
from src.inference import SentimentPredictor

predictor = SentimentPredictor(model_path='/opt/models/distilbert_sentiment')

def lambda_handler(event, context):
    text = event.get('body', {}).get('text', '')
    result = predictor.predict(text)
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
```

### Google Cloud Platform (GCP)

#### Cloud Run Deployment

```bash
# 1. Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT-ID/sentiment-analysis

# 2. Deploy to Cloud Run
gcloud run deploy sentiment-analysis \
  --image gcr.io/PROJECT-ID/sentiment-analysis \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

#### App Engine Deployment

```yaml
# app.yaml
runtime: python310
instance_class: F4

automatic_scaling:
  min_instances: 1
  max_instances: 10
  target_cpu_utilization: 0.65

env_variables:
  MODEL_PATH: 'models/distilbert_sentiment'
```

```bash
gcloud app deploy
```

### Microsoft Azure

#### Azure Container Instances

```bash
# Create resource group
az group create --name sentiment-rg --location eastus

# Deploy container
az container create \
  --resource-group sentiment-rg \
  --name sentiment-api \
  --image sentiment-analysis:latest \
  --cpu 2 \
  --memory 4 \
  --ports 5000 \
  --environment-variables \
    MODEL_PATH=/app/models/distilbert_sentiment
```

#### Azure App Service

```bash
# Create App Service plan
az appservice plan create \
  --name sentiment-plan \
  --resource-group sentiment-rg \
  --is-linux \
  --sku B2

# Create web app
az webapp create \
  --resource-group sentiment-rg \
  --plan sentiment-plan \
  --name sentiment-analysis-app \
  --deployment-container-image-name sentiment-analysis:latest
```

### DigitalOcean

#### Option 1: App Platform (Managed PaaS)

**Important**: The Docker image doesn't include model files (excluded by `.dockerignore`). You need to either:
- Build a custom image with models included, OR
- Modify your app to download models on startup from S3/cloud storage

**Option 1: Build image with models** (Quick but increases image size):

```dockerfile
# Add to Dockerfile before final CMD
COPY models/distilbert_sentiment/ /app/models/distilbert_sentiment/
# OR for quantized model:
# COPY models/distilbert_sentiment_quantized/ /app/models/distilbert_sentiment_quantized/
```

**Option 2: Download on startup** (Recommended for production):

Modify your app initialization to download models from cloud storage if not present locally.

**Deploy using DigitalOcean Container Registry:**

1. Push image to DigitalOcean Container Registry (see Method B below)

2. Create `app.yaml` in project root:

```yaml
name: sentiment-analysis
services:
  - name: api
    image:
      registry_type: DOCR
      repository: sentiment-registry/sentiment-analysis
      tag: latest
    instance_count: 1
    instance_size_slug: professional-xs
    http_port: 5000
    health_check:
      http_path: /health
      initial_delay_seconds: 60  # Allow time for model loading
    envs:
      - key: MODEL_PATH
        value: /app/models/distilbert_sentiment
        # Or use quantized model for faster inference:
        # value: /app/models/distilbert_sentiment_quantized
    routes:
      - path: /
```

2. Deploy via CLI:

```bash
# Install doctl
# Windows (PowerShell):
scoop install doctl
# or download from: https://github.com/digitalocean/doctl/releases

# macOS:
# brew install doctl

# Linux:
# cd ~
# wget https://github.com/digitalocean/doctl/releases/download/v1.94.0/doctl-1.94.0-linux-amd64.tar.gz
# tar xf ~/doctl-1.94.0-linux-amd64.tar.gz
# sudo mv ~/doctl /usr/local/bin

# Authenticate
doctl auth init

# Create app from spec
doctl apps create --spec app.yaml

# Get app ID and monitor deployment
doctl apps list
doctl apps logs <app-id> --follow
```

**Method B: Deploy from DigitalOcean Container Registry**

```bash
# 1. Create container registry
doctl registry create sentiment-registry

# 2. Log in to registry
doctl registry login

# 3. Build and tag image locally
docker build -t registry.digitalocean.com/sentiment-registry/sentiment-analysis:latest .

# 4. Push to registry
docker push registry.digitalocean.com/sentiment-registry/sentiment-analysis:latest

# 5. Deploy to App Platform using the app.yaml above
doctl apps create --spec app.yaml

# 6. Monitor deployment
doctl apps list
doctl apps logs <app-id> --follow

# 7. Get app URL
doctl apps get <app-id> --format DefaultIngress
```

> **Alternative**: You can also deploy via the DigitalOcean web console:
> 1. Go to Apps → Create App
> 2. Select "DigitalOcean Container Registry" as source
> 3. Choose your image: `sentiment-registry/sentiment-analysis:latest`
> 4. Configure resources (Professional-XS recommended for ML)
> 5. Deploy

#### Option 2: Kubernetes (DOKS)

```bash
# 1. Create Kubernetes cluster
doctl kubernetes cluster create sentiment-cluster \
  --region nyc3 \
  --node-pool "name=worker-pool;size=s-2vcpu-4gb;count=2"

# 2. Configure kubectl
doctl kubernetes cluster kubeconfig save sentiment-cluster

# 3. Create deployment manifest (deployment.yaml)
# Use your DigitalOcean Container Registry image, or alternatively GHCR if you push there
cat <<EOF > k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentiment-analysis
spec:
  replicas: 2
  selector:
    matchLabels:
      app: sentiment-analysis
  template:
    metadata:
      labels:
        app: sentiment-analysis
    spec:
      containers:
      - name: api
        image: registry.digitalocean.com/sentiment-registry/sentiment-analysis:latest
        # Alternative: use GHCR if you push there
        # image: ghcr.io/<your-username>/sentiment-analysis:latest
        ports:
        - containerPort: 5000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: sentiment-service
spec:
  type: LoadBalancer
  selector:
    app: sentiment-analysis
  ports:
  - port: 80
    targetPort: 5000
EOF

# 4. Apply manifests
kubectl apply -f k8s-deployment.yaml

# 5. Get external IP
kubectl get service sentiment-service
```

#### Option 3: Droplets with Docker

```bash
# 1. Create Droplet with Docker pre-installed
doctl compute droplet create sentiment-api \
  --image docker-20-04 \
  --size s-2vcpu-4gb \
  --region nyc3 \
  --ssh-keys <your-ssh-key-id>

# 2. SSH into droplet
doctl compute ssh sentiment-api

# 3. Build image on droplet (or pull from registry)
git clone https://github.com/<your-username>/sentiment-analysis.git
cd sentiment-analysis
docker build -t sentiment-analysis:latest .

# Alternative: pull from GHCR if you push there
# docker pull ghcr.io/<your-username>/sentiment-analysis:latest

# 4. Run container
docker run -d \
  -p 80:5000 \
  --name sentiment-api \
  --restart unless-stopped \
  sentiment-analysis:latest

# 4. Configure firewall (allow HTTP)
doctl compute firewall create \
  --name web-firewall \
  --inbound-rules "protocol:tcp,ports:80,sources:addresses:0.0.0.0/0,sources:addresses:::/0" \
  --droplet-ids $(doctl compute droplet list sentiment-api --format ID --no-header)
```

#### Option 4: Functions (Serverless)

For lightweight serverless deployment:

```python
# packages/sentiment/main.py
from src.inference import SentimentPredictor
import json

predictor = None

def main(args):
    global predictor
    if predictor is None:
        predictor = SentimentPredictor('models/distilbert_sentiment')
    
    text = args.get('text', '')
    result = predictor.predict(text)
    
    return {
        'body': json.dumps(result),
        'statusCode': 200
    }
```

Deploy:
```bash
doctl serverless deploy packages/sentiment
```

**Cost Comparison (DigitalOcean):**
- App Platform: $5-12/month (Professional-XS)
- Droplet: $18/month (2 vCPU, 4GB RAM)
- Kubernetes: ~$24/month (cluster + nodes)
- Functions: Pay per invocation (~$0.0000185 per GB-second)

## Production Considerations

### 1. Model Optimization

#### Quantization
```python
# Quantize model for faster inference
import torch

model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

#### ONNX Conversion
```python
# Convert to ONNX for optimized inference
from transformers import convert_graph_to_onnx

convert_graph_to_onnx.convert(
    framework="pt",
    model="models/distilbert_sentiment",
    output=Path("models/distilbert_onnx/model.onnx"),
    opset=12
)
```

### 2. Scaling

#### Horizontal Scaling
- Use load balancer (nginx, AWS ALB, GCP Load Balancer)
- Configure auto-scaling based on CPU/memory
- Implement health checks

#### Caching
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def predict_cached(text: str):
    return predictor.predict(text)
```

### 3. Monitoring

#### Application Metrics
- Request count
- Response time
- Error rate
- Model inference time

#### Tools
- Prometheus + Grafana
- AWS CloudWatch
- GCP Cloud Monitoring
- Azure Monitor

### 4. Security

```python
# Rate limiting - FastAPI (default)
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/predict")
@limiter.limit("50/hour")
async def predict(request: Request):
    pass

# Rate limiting - Flask alternative
# from flask_limiter import Limiter
#
# limiter = Limiter(
#     app,
#     default_limits=["200 per day", "50 per hour"]
# )

# API Key authentication
from functools import wraps

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != os.getenv('API_KEY'):
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function
```

### 5. Logging

```python
import logging
from pythonjsonlogger import jsonlogger

handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
handler.setFormatter(formatter)
logger.addHandler(handler)
```

### 6. CI/CD Pipeline

GitHub Actions (see `.github/workflows/ci.yaml`) provides:
- **Automated testing** across Python 3.10, 3.11, 3.12
- **Docker image building** (build verification only, no push):
  - GitHub Actions cache (`type=gha,mode=max`) for 50-80% faster rebuilds
  - Verifies Dockerfile builds correctly on every push to main
- **Build efficiency**: `.dockerignore` reduces context size significantly

> **Note**: The image (~3.5GB) is not pushed to a registry by default due to size. 
> To enable GHCR push, uncomment the relevant section in `ci.yaml`. 
> See [Using a Container Registry](#using-a-container-registry-optional) for manual push options.

### 7. Cost Optimization

#### AWS
- Use Spot Instances for training
- S3 for model storage
- Lambda for low-traffic APIs

#### GCP
- Use Preemptible VMs for training
- Cloud Storage for models
- Cloud Run for serverless deployment

#### General
- Implement request batching
- Use model quantization
- Cache frequent predictions
- Monitor and optimize resource usage

## Testing Deployment

```bash
# Health check
curl http://your-deployment-url/health

# Test prediction
curl -X POST http://your-deployment-url/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing!"}'

# Load testing
ab -n 1000 -c 10 -p test_payload.json \
  -T application/json \
  http://your-deployment-url/predict
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size, use quantized model
2. **Slow Inference**: Use GPU, optimize model, implement caching
3. **Container crashes**: Check resource limits, review logs
4. **Model not loading**: Verify model path, check file permissions

### Debug Commands

```bash
# Check container logs
docker logs sentiment-api

# Exec into container
docker exec -it sentiment-api bash

# Check resource usage
docker stats sentiment-api

# Test locally
python -c "from src.inference import SentimentPredictor; p = SentimentPredictor('models/distilbert_sentiment'); print(p.predict('test'))"
```
