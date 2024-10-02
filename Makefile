# Define model version
MAJOR=1
MINOR=0
PATCH=0
VERSION=$(MAJOR).$(MINOR).$(PATCH)

# Docker image variables
IMAGE_NAME=birehananteneh/loan-risk-predictor
TAG=$(VERSION)
PORT=5000
MODEL_PATH=mlruns/0/39cb1dffeba648cfb07c313ebd631ee7/artifacts/model


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
