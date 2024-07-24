import os
import shutil
import requests
import yaml
from git import Repo
from github import Github

# Load configuration from config.yaml
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

github_token = config["github_token"]
sonarqube_token = config["sonarqube_token"]
sonarqube_host = config["sonarqube_host"]
organization = config["organization"]
alm_setting = config["alm_setting"]
runs_on = config["runs_on"]

# Create a GitHub instance
gh = Github(github_token)

# Get the organization
org = gh.get_organization(organization)

# Iterate over repositories in the organization
for repo in org.get_repos():
    try:
        repo_name = repo.name
        print(repo_name)
        clone_url = repo.clone_url
        repo_path = f"./{repo_name}"

        # Remove the repository if it already exists locally
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)

        # Clone the repository
        Repo.clone_from(clone_url, repo_path)

        # Check if sonar-project.properties file exists
        properties_path = f"{repo_path}/sonar-project.properties"
        # if not os.path.isfile(properties_path):
        # Import GitHub project in SonarQube
        import_project_url = f"{sonarqube_host}/api/alm_integrations/import_github_project"
        headers = {
            "Authorization": f"Bearer {sonarqube_token}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {
            "almSetting": alm_setting,
            "organization": organization,
            "repositoryKey": f"{organization}/{repo_name}",
        }

        print("Creating SonarQube project...")
        response = requests.post(import_project_url, headers=headers, data=data)
        print(response.json())

        if response.status_code == 200:
            # Get the project key from the response
            project_key = response.json()["project"]["key"]

            print(project_key)
            # Create sonar-project.properties file
            with open(properties_path, "w") as file:
                file.write(f"sonar.projectKey={project_key}\n")
        else:
            print(f"Error importing GitHub project: {response.text}")

        # Create sonarqube.yml file in .github/workflows directory
        with open(f"{repo_path}/sonar-project.properties", "w") as file:
            file.write('''sonar.projectKey={project_key}'''.format(project_key=project_key))

        workflow_dir = f"{repo_path}/.github/workflows"
        os.makedirs(workflow_dir, exist_ok=True)
        workflow_file = f"{workflow_dir}/sonarqube.yml"
        with open(workflow_file, "w") as file:
            file.write('''name: SonarQube

on:
  workflow_dispatch:
  workflow_call:
  push:
    branches:
      - master

jobs:
  build:
    name: Sonarqube
    runs-on: [ {runs_on} ]
    steps:
      - uses: actions/checkout@v2
        with:
          fetch-depth: 0  # Shallow clones should be disabled for a better relevancy of analysis
      - uses: sonarsource/sonarqube-scan-action@master
        env:
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
          SONAR_HOST_URL: ${{ secrets.SONAR_HOST_URL }}
      # If you wish to fail your job when the Quality Gate is red, uncomment the
      # following lines. This would typically be used to fail a deployment.
      # - uses: sonarsource/sonarqube-quality-gate-action@master
      #   timeout-minutes: 5
      #   env:
      #     SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}\n'''.format(
                runs_on=runs_on,
            ))

        # Commit and push the changes
        repo = Repo(repo_path)
        index = repo.index
        index.add([
            ".github/workflows/sonarqube.yml",
            "sonar-project.properties"
        ])
        print("commit changes")
        index.commit("Add sonar-project.properties and sonarqube.yml")
        origin = repo.remote("origin")
        print("push changes")
        origin.push()

        # Remove the repository from the local machine
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)
    except Exception as e:
        print(e)
