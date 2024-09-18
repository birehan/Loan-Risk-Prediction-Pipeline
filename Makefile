# Define model version
MAJOR=1
MINOR=0
PATCH=0
VERSION=$(MAJOR).$(MINOR).$(PATCH)

# Docker image variables
IMAGE_NAME=birehananteneh/mlflow-model
TAG=$(VERSION)
PORT=5000
MODEL_PATH=notebooks/mlruns/0/ff624604f8eb44d59b1211e52f1fe564/artifacts/model
DOCKER_USERNAME=birehananteneh
DOCKER_PASSWORD=

# Log in to Docker Hub
login:
	echo "$(DOCKER_PASSWORD)" | docker login -u "$(DOCKER_USERNAME)" --password-stdin

# Build the Docker image with version tag
build:
	docker build --build-arg MODEL_PATH=$(MODEL_PATH) -t $(IMAGE_NAME):$(TAG) .

# Push the Docker image to Docker Hub
push: build
	docker push $(IMAGE_NAME):$(TAG)

# Run the Docker container
run:
	docker run -p $(PORT):$(PORT) $(IMAGE_NAME):$(TAG)

# Clean up unused Docker images
clean:
	docker rmi $(IMAGE_NAME):$(TAG)

# Help command
help:
	@echo "Makefile commands:"
	@echo "  make build   - Build the Docker image"
	@echo "  make push    - Build and push the Docker image to Docker Hub"
	@echo "  make run     - Run the Docker container"
	@echo "  make clean   - Remove the Docker image"
	@echo "  make help    - Display this help message"
