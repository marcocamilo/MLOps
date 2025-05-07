PROJECT_NAME ?= abc
MODEL ?= xyz
ENDPOINT_TYPE ?= api
DOCKERFILE ?= ./Dockerfile

# FIXED VARIABLES
PROJECT_ID ?= rnd-data-development
PROJECTS ?= rnd-data-development rnd-data-dw2
REGION ?= europe-west4
ARTIFACT_REPOSITORY ?= ml-containers
MODEL_NAME = $(PROJECT_NAME)-$(MODEL)
IMAGE_NAME = $(PROJECT_NAME)-docker
DOCKER_URI = europe-west3-docker.pkg.dev/$(PROJECT_ID)/$(ARTIFACT_REPOSITORY)/$(IMAGE_NAME):latest
ENDPOINT_NAME = $(PROJECT_NAME)-$(ENDPOINT_TYPE)-endpoint

SERVICE_ACCOUNT_NAME = $(PROJECT_NAME)-service-account
SERVICE_ACCOUNT_EMAIL = $(SERVICE_ACCOUNT_NAME)@$(PROJECT_ID).iam.gserviceaccount.com

# Optional parameters for deployment
MACHINE_TYPE ?= n1-standard-4
ACCELERATOR_TYPE ?= nvidia-tesla-t4
ACCELERATOR_COUNT ?= 2

# ──────────────────────────────────────────────────────────────────────
#   HELP TARGET                                                  
# ──────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo "ML Model Deployment Makefile"
	@echo "---------------------------"
	@echo "Available targets:"
	@echo "  make service-account   - Create service account & assign permissions"
	@echo "  make docker            - Build and push Docker container to Artifact Registry"
	@echo "  make register          - Register model to Vertex AI"
	@echo "  make create-endpoint   - Create a new endpoint (if not exists)"
	@echo "  make deploy            - Deploy model to an endpoint"
	@echo "  make undeploy          - Undeploy model from endpoint"
	@echo "  make delete-model      - Delete model from Vertex AI model registry"
	@echo "  make all               - Run all steps (service-account, docker, register, deploy)"
	@echo ""
	@echo "  make deploy-only       - Deploy model to existing endpoint (skips endpoint creation)"
	@echo "  make register-only     - Register model only (no docker build)"
	@echo "  make cleanup           - Undeploy model and delete it from registry"

# ──────────────────────────────────────────────────────────────────────
#   Step 0: Create Service Account & Assign Permissions                
# ──────────────────────────────────────────────────────────────────────
.PHONY: service-account
service-account:
	@echo "🔧 Creating service account for deployment..."
	gcloud iam service-accounts create $(SERVICE_ACCOUNT_NAME) \
	  --project=$(PROJECT_ID) \
	  --description="Service account for Vertex AI model deployment" \
	  --display-name="$(PROJECT_NAME) Service Account"
	
	@for PROJECT in $(PROJECTS); do \
		echo "➡️ Assigning permissions in project: $$PROJECT"; \
		gcloud projects add-iam-policy-binding $$PROJECT \
		  --member="serviceAccount:$(SERVICE_ACCOUNT_EMAIL)" \
		  --role="roles/bigquery.admin"; \
		gcloud projects add-iam-policy-binding $$PROJECT \
		  --member="serviceAccount:$(SERVICE_ACCOUNT_EMAIL)" \
		  --role="roles/bigquery.dataViewer"; \
		gcloud projects add-iam-policy-binding $$PROJECT \
		  --member="serviceAccount:$(SERVICE_ACCOUNT_EMAIL)" \
		  --role="roles/bigquery.jobUser"; \
		gcloud projects add-iam-policy-binding $$PROJECT \
		  --member="serviceAccount:$(SERVICE_ACCOUNT_EMAIL)" \
		  --role="roles/serviceusage.serviceUsageConsumer"; \
		echo "✅ Permissions assigned for $$PROJECT"; \
	done
	
	@echo "➡️ Assigning permissions in bucket: prod-data-etl-arcpublishing"
	gcloud storage buckets add-iam-policy-binding "gs://prod-data-etl-arcpublishing" \
	  --member="serviceAccount:$(SERVICE_ACCOUNT_EMAIL)" \
	  --role="roles/storage.objectViewer"
	
	@echo "🔽 Generating and downloading service key..."
	gcloud iam service-accounts keys create $(SERVICE_ACCOUNT_NAME).json \
	  --iam-account $(SERVICE_ACCOUNT_EMAIL)
	
	@echo "✅ Permissions assigned and key saved."

# ──────────────────────────────────────────────────────────────────────
#   Step 1: Storing Docker container to Artifact Registry              
# ──────────────────────────────────────────────────────────────────────
.PHONY: docker
docker:
	@echo "📤 Storing Docker container to Artifact Registry..."
	docker build -t $(DOCKER_URI) -f $(DOCKERFILE) .
	docker push $(DOCKER_URI)
	@echo "✅ Docker container pushed successfully to Artifact Registry."

# ──────────────────────────────────────────────────────────────────────
#   Step 2: Register model to Vertex AI                                
# ──────────────────────────────────────────────────────────────────────
.PHONY: register register-only
register: docker register-only

register-only:
	@echo "🔄 Importing custom container to Model Registry..."
	gcloud ai models upload \
	  --region=$(REGION) \
	  --project=$(PROJECT_ID) \
	  --display-name=$(MODEL_NAME) \
	  --container-image-uri=$(DOCKER_URI) \
	  --container-predict-route="/predict" \
	  --container-health-route="/health"
	@echo "✅ Model imported successfully into Vertex AI."

# ──────────────────────────────────────────────────────────────────────
#   Step 3: Deploy the model to an endpoint                            
# ──────────────────────────────────────────────────────────────────────
.PHONY: create-endpoint 
create-endpoint:
	@echo "🚀 Creating endpoint in Vertex AI..."
	-gcloud ai endpoints create \
	  --region=$(REGION) \
	  --project=$(PROJECT_ID) \
	  --display-name=$(ENDPOINT_NAME)
	@echo "✅ Endpoint created (or already exists)."

.PHONY: deploy deploy-only
deploy: create-endpoint deploy-only

deploy-only:
	@echo "🚀 Deploying model to endpoint in Vertex AI..."
	$(eval ENDPOINT_ID=$(shell gcloud ai endpoints list \
	  --region=$(REGION) \
	  --project=$(PROJECT_ID) \
	  --filter="displayName=$(ENDPOINT_NAME)" \
	  --format="value(name)"))
	
	$(eval MODEL_ID=$(shell gcloud ai models list \
	  --region=$(REGION) \
	  --project=$(PROJECT_ID) \
	  --filter="displayName=$(MODEL_NAME)" \
	  --format="value(name)"))
	
	@echo "Using Endpoint ID: $(ENDPOINT_ID)"
	@echo "Using Model ID: $(MODEL_ID)"
	
	gcloud ai endpoints deploy-model $(ENDPOINT_ID) \
	  --region=$(REGION) \
	  --project=$(PROJECT_ID) \
	  --model=$(MODEL_ID) \
	  --display-name=$(MODEL_NAME) \
	  --machine-type="$(MACHINE_TYPE)" \
	  --accelerator=type=$(ACCELERATOR_TYPE),count=$(ACCELERATOR_COUNT) \
	  --enable-access-logging \
	  --service-account=$(SERVICE_ACCOUNT_EMAIL)
	
	@echo "✅ Model deployed to endpoint!"

# ──────────────────────────────────────────────────────────────────────
#   REMOVE MODEL                                                    
# ──────────────────────────────────────────────────────────────────────
.PHONY: undeploy
undeploy:
	@echo "🛑 Undeploying model from endpoint..."
	$(eval ENDPOINT_ID=$(shell gcloud ai endpoints list \
	  --region=$(REGION) \
	  --project=$(PROJECT_ID) \
	  --filter="displayName=$(ENDPOINT_NAME)" \
	  --format="value(name)"))
	
	$(eval DEPLOYED_MODEL_ID=$(shell gcloud ai endpoints describe $(ENDPOINT_ID) \
	  --region=$(REGION) \
	  --project=$(PROJECT_ID) \
	  --format="value(deployedModels[0].id)"))
	
	@echo "Using Endpoint ID: $(ENDPOINT_ID)"
	@echo "Using Deployed Model ID: $(DEPLOYED_MODEL_ID)"
	
	gcloud ai endpoints undeploy-model $(ENDPOINT_ID) \
	  --region=$(REGION) \
	  --deployed-model-id=$(DEPLOYED_MODEL_ID)
	
	@echo "✅ Model undeployed from endpoint!"

# ──────────────────────────────────────────────────────────────────────
#   DELETE MODEL FROM REGISTRY                                                    
# ──────────────────────────────────────────────────────────────────────
.PHONY: delete-model
delete-model:
	@echo "🗑️ Deleting model from Vertex AI model registry..."
	$(eval MODEL_ID=$(shell gcloud ai models list \
	  --region=$(REGION) \
	  --project=$(PROJECT_ID) \
	  --filter="displayName=$(MODEL_NAME)" \
	  --format="value(name)"))
	
	@echo "Using Model ID: $(MODEL_ID)"
	
	gcloud ai models delete $(MODEL_ID) \
	  --region=$(REGION) \
	  --project=$(PROJECT_ID) \
	  --quiet
	
	@echo "✅ Model deleted from registry!"

# ──────────────────────────────────────────────────────────────────────
#   CLEANUP (UNDEPLOY + DELETE)                                                    
# ──────────────────────────────────────────────────────────────────────
.PHONY: cleanup
cleanup: undeploy delete-model
	@echo "✅ Complete cleanup finished - model undeployed and deleted from registry!"

# ──────────────────────────────────────────────────────────────────────
#   ALL STEPS                                                    
# ──────────────────────────────────────────────────────────────────────
.PHONY: all
all: service-account docker register deploy
