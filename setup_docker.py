import subprocess
import sys

def run_docker_compose():
    """Run docker-compose commands to set up the Docker containers."""
    print("Starting Docker containers...")
    try:
        subprocess.run(["docker", "compose", "build"], check=True)
        subprocess.run(["docker", "compose", "up", "-d"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Docker setup failed with error: {e}")
        sys.exit(e.returncode)

def main():
    print("Running Docker setup...")
    run_docker_compose()
    print("Docker containers are now running.")

if __name__ == "__main__":
    main()
