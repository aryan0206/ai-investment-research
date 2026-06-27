from pathlib import Path
from typing import List
from langchain_core.documents import Document
from pypdf import PdfReader


class PDFParser:
    def __init__(self, reports_dir: str = "./data/reports"):
        self.reports_dir = Path(reports_dir)

    def parse(self, filename: str) -> List[Document]:
        """Extract and chunk text from a single annual report PDF."""
        path = self.reports_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Report not found: {path}")

        reader = PdfReader(str(path))
        documents = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                documents.append(
                    Document(
                        page_content=text.strip(),
                        metadata={
                            "source": filename,
                            "page": i + 1,
                            "type": "annual_report",
                        },
                    )
                )

        print(f"[PDFParser] Parsed {len(documents)} pages from {filename}")
        return documents

    def parse_all(self) -> List[Document]:
        """Parse all PDFs found in the reports directory."""
        all_docs = []
        pdf_files = list(self.reports_dir.glob("*.pdf"))

        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {self.reports_dir}")

        for pdf_path in pdf_files:
            all_docs.extend(self.parse(pdf_path.name))

        print(f"[PDFParser] Total pages across all reports: {len(all_docs)}")
        return all_docs
