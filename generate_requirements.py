import subprocess

# Open requirements.txt in write mode with UTF-8 encoding
with open("requirements.txt", "w", encoding="utf-8") as requirements_file:
    # Run pip freeze and write output directly to the file
    subprocess.run(["python", "-m", "pip", "freeze"], stdout=requirements_file)

print("requirements.txt created successfully in UTF-8!")
