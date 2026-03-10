import os
import pandas as pd


def extract_email_data(directory_path):
    extracted_data = []

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)

            file_date = ""
            file_from = ""
            file_to = ""

            try:
                with open(file_path, mode='r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if line.startswith("Date: ") and not file_date:
                            file_date = line.replace("Date: ", "").strip()
                        elif line.startswith("From: ") and not file_from:
                            file_from = line.replace("From: ", "").strip()
                        elif line.startswith("To: ") and not file_to:
                            file_to = line.replace("To: ", "").strip()

                        if file_date and file_from and file_to:
                            break

                if file_date or file_from or file_to:
                    extracted_data.append({
                        'File Path': file_path,
                        'Date': file_date,
                        'From': file_from,
                        'To': file_to
                    })
                else:
                    print(f"Date, From, and To not found in file: {file_path}")

            except Exception as e:
                print(f"Could not read {file_path}. Error: {e}")

    df = pd.DataFrame(extracted_data)
    return df


target_directory = "../data/maildir"
output_filename = "extracted_data.csv"

email_df = extract_email_data(target_directory)
email_df.to_csv(output_filename, index=False)
