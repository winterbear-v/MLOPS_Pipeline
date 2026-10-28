pipeline {
    agent any

    environment {
        COMPOSE_PROJECT_NAME = "model_factory"
    }

    stages {
        stage('1. Checkout & Verify Line Endings') {
            steps {
                checkout scm
                sh 'git status'
            }
        }

        stage('2. Unit & Privacy/Pipeline Tests (Pytest)') {
            steps {
                sh '''
                    python3 -m venv .ci_venv
                    . .ci_venv/bin/activate
                    pip install --upgrade pip
                    pip install -r tests/requirements.txt
                    pytest tests/ -v
                '''
            }
        }

        stage('3. Build Base Serving Image & Platform Containers') {
            steps {
                sh '''
                    docker build -t model-factory-base-server:latest ./base_model_server
                    docker compose build
                '''
            }
        }

        stage('4. Deploy Platform Stack & End-to-End Smoke Check') {
            steps {
                sh '''
                    docker compose up -d
                    sleep 10
                    curl -f http://localhost:5001/health
                    curl -f http://localhost:5002/health
                '''
            }
        }
    }

    post {
        always {
            sh 'rm -rf .ci_venv || true'
        }
    }
}
