IMAGE_NAME=xyz
GCP_PROJECT=$(PROJECT_ID)
ARTIFACT_REGISTRY_NAME=repo
ARTIFACT_REGION=$(REGION)
VERSION=latest

build:
	docker build -t "${ARTIFACT_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACT_REGISTRY_NAME}/${IMAGE_NAME}:${VERSION}" .
	
docker-auth:
	gcloud auth configure-docker ${ARTIFACT_REGION}-docker.pkg.dev

push:
	docker push "${ARTIFACT_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACT_REGISTRY_NAME}/${IMAGE_NAME}:${VERSION}"

all: build push

