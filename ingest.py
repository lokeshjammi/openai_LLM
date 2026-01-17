import glob

knowledge_base_path = "solar_knowledge_base/**/*.md"
files = glob.glob(knowledge_base_path)

print(files)