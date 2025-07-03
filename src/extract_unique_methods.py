import os
import glob
import multiprocessing

def extract_methods_from_file(file_path):
    methods = set()
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split(' | ', 3)
            if len(parts) >= 3:
                methods.add(parts[2])
    return methods

def extract_methods_from_files(log_dir, pattern="*.log", num_workers=None):
    log_files = glob.glob(os.path.join(log_dir, pattern))

    if num_workers is None:
        num_workers = min(len(log_files), multiprocessing.cpu_count())

    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(extract_methods_from_file, log_files)

    all_methods = set()
    for methods in results:
        all_methods.update(methods)

    return all_methods

def write_methods_to_file(methods, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for method in sorted(methods):
            f.write(method + '\n')

if __name__ == "__main__":
    log_directory = "../data/CCI"      # 🔁 Change to your actual log directory
    output_file = "unique_methods.txt"   # 🔁 Change if you want a different output name

    method_set = extract_methods_from_files(log_directory)
    write_methods_to_file(method_set, output_file)

    print(f"Wrote {len(method_set)} unique methods to: {output_file}")
