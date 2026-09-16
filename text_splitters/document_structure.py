from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language
)


# Python code
code = """
def calculate_sum(a, b):
    result = a + b
    return result


def calculate_average(numbers):
    total = sum(numbers)
    average = total / len(numbers)
    return average


def main():
    numbers = [10, 20, 30, 40, 50]

    total = calculate_sum(numbers[0], numbers[1])

    average = calculate_average(numbers)

    print("Total:", total)
    print("Average:", average)


if __name__ == "__main__":
    main()
"""


# Initialize the splitter
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=300,
    chunk_overlap=0,
)


# Split the code
chunks = splitter.split_text(code)


# Print chunks
print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n========== CHUNK {i + 1} ==========")
    print(chunk)