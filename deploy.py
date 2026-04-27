#!/usr/bin/env python3
"""
Palmora Deployment Helper
Automates deployment to various cloud platforms
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def run_command(cmd, description=""):
    """Run a shell command with error handling"""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}: {e.stderr}")
        return None

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking deployment requirements...")
    
    # Check if git is installed
    if run_command("git --version", "Checking git installation") is None:
        print("❌ Git is required for deployment")
        return False
    
    # Check if we're in a git repository
    if run_command("git status", "Checking git repository") is None:
        print("❌ Not in a git repository. Run 'git init' first.")
        return False
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY environment variable not set")
        print("   You'll need to configure this in your deployment platform")
    
    print("✅ Requirements check completed")
    return True

def deploy_streamlit_cloud():
    """Deploy to Streamlit Cloud"""
    print("\n🚀 Deploying to Streamlit Cloud")
    print("=" * 50)
    
    print("📝 Steps to deploy on Streamlit Cloud:")
    print("1. Push your code to GitHub:")
    print("   git add .")
    print("   git commit -m 'Ready for deployment'")
    print("   git push origin main")
    print()
    print("2. Go to https://share.streamlit.io")
    print("3. Connect your GitHub account")
    print("4. Select your repository and branch")
    print("5. Set main file path: app.py")
    print("6. Add secrets in the dashboard:")
    print("   OPENAI_API_KEY = your_api_key_here")
    print("   PALM_READER_VISION_MODEL = gpt-4o")
    print("   PALM_READER_IMAGE_MODEL = dall-e-3")
    print()
    
    # Auto-push to git if user wants
    push = input("🤔 Push to GitHub now? (y/n): ").lower().strip()
    if push == 'y':
        run_command("git add .", "Adding files to git")
        commit_msg = input("📝 Enter commit message (or press Enter for default): ").strip()
        if not commit_msg:
            commit_msg = "Deploy Palmora to web"
        
        run_command(f'git commit -m "{commit_msg}"', "Committing changes")
        run_command("git push origin main", "Pushing to GitHub")
        print("✅ Code pushed to GitHub!")
        print("🌐 Now go to https://share.streamlit.io to complete deployment")

def deploy_docker():
    """Deploy using Docker"""
    print("\n🐳 Deploying with Docker")
    print("=" * 50)
    
    # Build Docker image
    if run_command("docker build -t palmora .", "Building Docker image"):
        print("✅ Docker image built successfully")
        
        # Run container
        api_key = input("🔑 Enter your OpenAI API key: ").strip()
        if api_key:
            cmd = f'docker run -d -p 8501:8501 -e OPENAI_API_KEY="{api_key}" palmora'
            if run_command(cmd, "Starting Docker container"):
                print("🎉 Palmora is now running at http://localhost:8501")
        else:
            print("⚠️  No API key provided. Set OPENAI_API_KEY environment variable.")

def deploy_heroku():
    """Deploy to Heroku"""
    print("\n🟣 Deploying to Heroku")
    print("=" * 50)
    
    # Copy Heroku files to root
    os.system("cp deploy/heroku/* .")
    
    print("📝 Steps for Heroku deployment:")
    print("1. Install Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli")
    print("2. Login: heroku login")
    print("3. Create app: heroku create your-app-name")
    print("4. Set config: heroku config:set OPENAI_API_KEY=your_key")
    print("5. Deploy: git push heroku main")

def deploy_railway():
    """Deploy to Railway"""
    print("\n🚂 Deploying to Railway")
    print("=" * 50)
    
    print("📝 Steps for Railway deployment:")
    print("1. Go to https://railway.app")
    print("2. Connect your GitHub repository")
    print("3. Add environment variables:")
    print("   OPENAI_API_KEY = your_api_key_here")
    print("4. Deploy automatically from GitHub")

def main():
    """Main deployment function"""
    print("🔮 Palmora Deployment Helper")
    print("=" * 50)
    
    if not check_requirements():
        print("❌ Requirements not met. Please fix the issues above.")
        sys.exit(1)
    
    print("\n🌐 Available deployment options:")
    print("1. Streamlit Cloud (Free, Recommended)")
    print("2. Docker (Local/Cloud)")
    print("3. Heroku")
    print("4. Railway")
    print("5. Exit")
    
    while True:
        choice = input("\n🤔 Select deployment option (1-5): ").strip()
        
        if choice == "1":
            deploy_streamlit_cloud()
            break
        elif choice == "2":
            deploy_docker()
            break
        elif choice == "3":
            deploy_heroku()
            break
        elif choice == "4":
            deploy_railway()
            break
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1-5.")

if __name__ == "__main__":
    main()