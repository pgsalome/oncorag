#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path
import time
from dotenv import load_dotenv

def create_virtualenv():
    """Create a virtual environment if it doesn't already exist."""
    venv_path = Path("venv")
    if not venv_path.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created successfully.")
    else:
        print("Virtual environment already exists.")
    return venv_path

def install_requirements():
    """Install dependencies from requirements.txt into the virtual environment."""
    requirements_path = Path("requirements.txt")

    # Determine the correct Python interpreter path based on the operating system
    if sys.platform == "win32":
        python_path = Path("venv/Scripts/python")
    else:  # Unix-based systems (Linux, macOS)
        python_path = Path("venv/bin/python")

    if not python_path.exists():
        print(f"❌ Virtual environment Python not found at {python_path}!")
        sys.exit(1)

    if requirements_path.exists():
        print("Installing dependencies from requirements.txt into the virtual environment...")
        # Use the virtual environment's Python to run pip, ensuring packages go into the venv
        subprocess.run([str(python_path), "-m", "pip", "install", "-r", str(requirements_path)], check=True)
        print("✅ Dependencies installed successfully in the virtual environment.")
    else:
        print("⚠️ requirements.txt not found. Skipping Python dependency installation.")

def generate_env_file():
    """Generate the .env file if it doesn't exist."""
    env_file_path = '.env'
    if os.path.exists(env_file_path):
        print(f"Found existing .env file at {env_file_path}. Using the values from it.")
        load_dotenv()  # Load environment variables from the .env file
        return  # If .env exists, just return and use those values

    print("No .env file found. Please provide the following values:")
    # API Keys and configuration
    openai_api_key = input("Enter your OpenAI API key: ")
    # Add all other keys as needed...

    # Write the values to the .env file
    with open(env_file_path, "w") as env_file:
        env_file.write(f"""
OPENAI_API_KEY={openai_api_key}
# Add other API keys here
""")
    print(f"✅ .env file has been created successfully at {env_file_path}!")
    load_dotenv()  # Load the environment variables after writing the .env file

def main():
    # Step 1: Create the virtual environment
    venv_path = create_virtualenv()

    # Step 2: Install project dependencies
    # We use the virtual environment's Python interpreter to run pip
    # This ensures packages are installed in the virtual environment
    install_requirements()

    # Step 3: Generate or read the .env file
    generate_env_file()

    # Print instructions on how to activate the virtual environment
    if sys.platform == "win32":
        activate_cmd = ".\\venv\\Scripts\\activate"
    else:  # Unix-based systems (Linux, macOS)
        activate_cmd = "source venv/bin/activate"

    print("\n✅ Environment setup is complete!")
    print(f"\nTo activate the virtual environment, run:\n\n    {activate_cmd}\n")

if __name__ == "__main__":
    main()