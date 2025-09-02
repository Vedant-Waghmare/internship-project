from FileReader import FileReader
from JobParser import JobParser
from CSVExporter import CSVExporter

def main():
    file_path = "data/sample_job.txt"

    # Reading
    reader = FileReader(file_path)
    raw_text = reader.read()
    print("File read successfully")
    print(raw_text, "\n") 

    # Parseing
    parser = JobParser(raw_text)
    job_data = parser.extract_features()
    print("Parsing completed")

    # Exporting
    exporter = CSVExporter("outputs/jobs.csv")
    exporter.save([job_data])
    print("Exported to outputs/jobs.csv successfully")

if __name__ == "__main__":
    main()
