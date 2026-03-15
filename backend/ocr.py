"""OCR utilities for scanned PDFs and standalone images."""

import logging
from pathlib import Path

import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from langchain.schema import Document

logger = logging.getLogger("brainllm.ocr")

MIN_CHARS_PER_PAGE = 50

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


def is_scanned_pdf(docs: list[Document], page_count: int) -> bool:
    """Return True if the PDF appears to be scanned (little/no embedded text).

    Heuristic: if the average characters per page falls below
    MIN_CHARS_PER_PAGE the file is almost certainly a scan or
    image-only PDF.
    """
    if page_count == 0:
        return True

    total_chars = sum(len(doc.page_content.strip()) for doc in docs)
    avg = total_chars / page_count

    if avg < MIN_CHARS_PER_PAGE:
        logger.info(
            "PDF detected as scanned (avg %.1f chars/page, threshold %d)",
            avg,
            MIN_CHARS_PER_PAGE,
        )
        return True

    return False


def ocr_pdf(pdf_path: str) -> list[Document]:
    """OCR a scanned PDF by converting pages to images and running Tesseract.

    Each page is processed independently so a single bad page doesn't
    crash the whole file.
    """
    logger.info("Running OCR on PDF: %s", pdf_path)

    try:
        images = convert_from_path(pdf_path, dpi=300)
    except Exception as e:
        logger.error("Failed to convert PDF to images: %s — %s", pdf_path, e)
        return []

    docs: list[Document] = []
    for page_num, image in enumerate(images, start=1):
        try:
            text = pytesseract.image_to_string(image)
            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path,
                            "page": page_num,
                            "ocr": True,
                        },
                    )
                )
        except Exception as e:
            logger.error("OCR failed on page %d of %s: %s", page_num, pdf_path, e)

    logger.info("OCR extracted text from %d/%d pages", len(docs), len(images))
    return docs


def ocr_image(image_path: str) -> list[Document]:
    """OCR a standalone image file."""
    logger.info("Running OCR on image: %s", image_path)

    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
    except Exception as e:
        logger.error("OCR failed on image %s: %s", image_path, e)
        return []

    if not text.strip():
        logger.warning("No text extracted from image: %s", image_path)
        return []

    return [
        Document(
            page_content=text,
            metadata={"source": image_path, "ocr": True},
        )
    ]
