import pdfplumber
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import os


# Predefined that the logical rows are handling the case of header keywords being on two lines



HEADER_KEYWORDS = [
    # Serial/Number variations
    "sr no", "srno", "sr.no.", "serial", "s.no", "serial number", "no.", "srl",
    "sl. no.", "#",

    # Date variations
    "date", "transaction date", "trans date", "trans dt", "trn dt", "txn date",
    "date of transaction", "payment date", "entry date", "tran date", "trn. date",
    "completion time", "date id", "post date", "value date", "val date", "value dt", "value",

    # Description variations
    "description", "desc", "particulars", "details", "narration", "narrative",
    "transaction details", "remarks", "transaction description", "details of transaction",

    # Amount variations
    "amount", "amt", "value", "transaction amount",

    # Transaction type variations
    "transaction type", "txn type", "type", "credits/debits", "cr/dr", "credit/debit",
    "credit or debit", "dr / cr", "withdrawal (dr)/deposit (cr)", "debit/credit", "debit/cr", "debit/credit",
    "debit/credit", "withdrawal (dr)/deposit (cr)", "debit/credit", "debit", "credit",

    # Debit variations
    "debit", "withdrawal", "payment", "dr", "dr amount", "dr.", "withdrawl(dr)",
    "withdrawals", "debits", "withdrawal amt.", "transaction debit amount", "debit amt",
    "paid in", "withdraw(dr amount)",

    # Credit variations
    "credit", "deposit", "receipt", "cr", "cr amount", "cr.", "deposit(cr)",
    "deposits", "credits", "deposit amt.", "transaction credit amount", "credit amt",
    "withdrawn", "deposit(cr amount)",

    # Beneficiary / payee variations
    "beneficiary", "payee", "dealer name", "name", "party", "recipient",

    # Reference number variations
    "ref", "ref no", "reference", "customer ref", "cust ref no", "transaction id",
    "utr", "cheque/ref", "chq/ref no", "cheque number", "chq no", "chq/ref", "chq.",
    "ref.no", "chq.no.", "chq-no", "chq/ref.no", "cheque no.", "chq./req. number",
    "cheque/ref.no.", "chq./ref.no.", "chequeno.", "chq. no.", "chq no/ref no",
    "chq./ref. number", "ref. no", "chq / ref number", "cheque/reference#", "chq / ref no.",
    "cheque#", "receipt no", "ref no./cheque no.", "cheque no/ reference no",
    "cheque no/reference no", "ref num", "utr number", "utr", "transaction id",
    "tran id", "instrument no", "instrument number", "inst no", "inst number",
    "instr. no.", "instr no", "instruments", "instrmnt number", "instrument id",

    # Balance variations
    "balance", "bal", "closing balance", "running balance", "available balance",
    "available bal.", "closing bal", "total amount dr/cr", "total amount", "balance amt",
]

# Load sentence transformer model once
transformer_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
header_keyword_embeddings = transformer_model.encode(HEADER_KEYWORDS, convert_to_tensor=True)


def group_chars_to_lines(chars, vertical_tol=0):
    if chars.empty:
        return []
    chars = chars.sort_values(by=["y0", "x0"]).to_dict("records")
    lines = []
    for char in chars:
        placed = False
        for line in lines:
            line_y0s = [c['y0'] for c in line]
            line_y1s = [c['y1'] for c in line]
            line_min_y0 = min(line_y0s)
            line_max_y1 = max(line_y1s)
            if not (char['y1'] + vertical_tol < line_min_y0 or char['y0'] - vertical_tol > line_max_y1):
                line.append(char)
                placed = True
                break
        if not placed:
            lines.append([char])
    for line in lines:
        line.sort(key=lambda c: c['x0'])
    return lines


def group_lines_to_blocks(lines, gap_tol=10):
    line_positions = []
    for line in lines:
        ys0 = [c['y0'] for c in line]
        ys1 = [c['y1'] for c in line]
        avg_y0 = np.mean(ys0)
        avg_y1 = np.mean(ys1)
        line_positions.append((avg_y0, avg_y1))

    blocks = []
    current_block = [lines[0]]
    current_block_bottom = line_positions[0][1]

    for i in range(1, len(lines)):
        this_line_top = line_positions[i][0]
        if (this_line_top - current_block_bottom) <= gap_tol:
            current_block.append(lines[i])
            current_block_bottom = max(current_block_bottom, line_positions[i][1])
        else:
            blocks.append(current_block)
            current_block = [lines[i]]
            current_block_bottom = line_positions[i][1]

    if current_block:
        blocks.append(current_block)
    return blocks


def line_to_text_segments(line_chars, horizontal_gap_thresh=8):
    if not line_chars:
        return []
    segments = []
    current_segment_chars = [line_chars[0]]
    for prev_char, char in zip(line_chars[:-1], line_chars[1:]):
        gap = char['x0'] - prev_char['x1']
        if gap > horizontal_gap_thresh:
            segment_text = "".join(c['text'] for c in current_segment_chars).strip()
            if segment_text:
                segments.append(segment_text)
            current_segment_chars = [char]
        else:
            current_segment_chars.append(char)
    segment_text = "".join(c['text'] for c in current_segment_chars).strip()
    if segment_text:
        segments.append(segment_text)
    return segments


def compute_semantic_header_score(block_segments):
    """
    Given a multi-line block (list of lines, each a list of text segments),
    compute semantic similarity to header keywords by averaging max similarity
    of each text segment in the block.
    """
    block_texts = [seg for line in block_segments for seg in line if seg]  # flatten and remove empty
    if not block_texts:
        return 0.0
    block_embeddings = transformer_model.encode(block_texts, convert_to_tensor=True)
    scores = []
    for emb in block_embeddings:
        similarities = util.pytorch_cos_sim(emb, header_keyword_embeddings)[0]
        max_sim, _ = torch.max(similarities, 0)
        scores.append(max_sim.item())
    return np.mean(scores) if scores else 0.0


def extract_header_blocks_with_confidence(pdf_path, page_num=0):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        chars = pd.DataFrame(page.chars)
        if chars.empty:
            return []

        lines = group_chars_to_lines(chars, vertical_tol=0)
        blocks = group_lines_to_blocks(lines, gap_tol=10)

        header_blocks = []
        for block in blocks:
            block_segments = [line_to_text_segments(line, horizontal_gap_thresh=8) for line in block]
            confidence = compute_semantic_header_score(block_segments)
            header_blocks.append((block_segments, confidence))

        # Sort blocks by confidence descending
        header_blocks.sort(key=lambda x: x[1], reverse=True)
        return header_blocks


if __name__ == "__main__":
    pdf_folder = "../../tests/Priority Banks"
    output_folder = "./header_detection_results"
    os.makedirs(output_folder, exist_ok=True)

    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        try:
            header_blocks_with_conf = extract_header_blocks_with_confidence(pdf_path, page_num=0)
            top_5 = header_blocks_with_conf[:5]

            output_text = f"PDF: {pdf_file}\nTop 5 detected header blocks with confidence scores:\n"
            print(f"Processing {pdf_file}...")
            print(output_text)

            for i, (block_segments, confidence) in enumerate(top_5):
                output_text += f"Block {i} confidence: {confidence:.4f}\n"
                print(f"Block {i} confidence: {confidence:.4f}")
                for line_segments in block_segments:
                    segment_str = ", ".join(line_segments)
                    output_text += f"  Line segments: {segment_str}\n"
                    print(f"  Line segments: {segment_str}")

            output_text += "\n"

            output_file_path = os.path.join(output_folder, f"{pdf_file.replace(' ', '_').replace('.pdf', '')}_headers.txt")
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(output_text)

            print(f"Results saved to {output_file_path}\n")

        except Exception as e:
            print(f"Error processing {pdf_file}: {e}\n")

