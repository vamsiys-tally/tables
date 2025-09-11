import pdfplumber
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import os

HEADER_KEYWORDS = [
    "sr no", "srno", "sr.no.", "serial", "s.no", "serial number", "no.", "srl",
    "sl. no.", "#", "date", "transaction date", "trans date", "trans dt", "trn dt", "txn date",
    "date of transaction", "payment date", "entry date", "tran date", "trn. date",
    "completion time", "date id", "post date", "value date", "val date", "value dt", "value",
    "description", "desc", "particulars", "details", "narration", "narrative",
    "transaction details", "remarks", "transaction description", "details of transaction",
    "amount", "amt", "value", "transaction amount",
    "transaction type", "txn type", "type", "credits/debits", "cr/dr", "credit/debit",
    "credit or debit", "dr / cr", "withdrawal (dr)/deposit (cr)", "debit/credit", "debit/cr", "debit/credit",
    "debit/credit", "withdrawal (dr)/deposit (cr)", "debit/credit", "debit", "credit",
    "debit", "withdrawal", "payment", "dr", "dr amount", "dr.", "withdrawl(dr)",
    "withdrawals", "debits", "withdrawal amt.", "transaction debit amount", "debit amt",
    "paid in", "withdraw(dr amount)",
    "credit", "deposit", "receipt", "cr", "cr amount", "cr.", "deposit(cr)",
    "deposits", "credits", "deposit amt.", "transaction credit amount", "credit amt",
    "withdrawn", "deposit(cr amount)",
    "beneficiary", "payee", "dealer name", "name", "party", "recipient",
    "ref", "ref no", "reference", "customer ref", "cust ref no", "transaction id",
    "utr", "cheque/ref", "chq/ref no", "cheque number", "chq no", "chq/ref", "chq.",
    "ref.no", "chq.no.", "chq-no", "chq/ref.no", "cheque no.", "chq./req. number",
    "cheque/ref.no.", "chq./ref.no.", "chequeno.", "chq. no.", "chq no/ref no",
    "chq./ref. number", "ref. no", "chq / ref number", "cheque/reference#", "chq / ref no.",
    "cheque#", "receipt no", "ref no./cheque no.", "cheque no/ reference no",
    "cheque no/reference no", "ref num", "utr number", "utr", "transaction id",
    "tran id", "instrument no", "instrument number", "inst no", "inst number",
    "instr. no.", "instr no", "instruments", "instrmnt number", "instrument id",
    "balance", "bal", "closing balance", "running balance", "available balance",
    "available bal.", "closing bal", "total amount dr/cr", "total amount", "balance amt",
]

COMMON_HEADERS = {"CR", "DR"}

transformer_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
header_keyword_embeddings = transformer_model.encode(HEADER_KEYWORDS, convert_to_tensor=True)

def extract_horizontal_edges(page, tol=2.0):
    """Extract horizontal edges from page lines and edges"""
    lines = pd.DataFrame(page.lines + page.edges if hasattr(page, 'edges') else page.lines)
    if lines.empty:
        return []
    lines['orientation'] = lines.apply(lambda row: 'horizontal' 
                                      if np.isclose(row['y0'], row['y1'], atol=tol) else 'other', axis=1)
    hor_lines = lines[lines['orientation'] == 'horizontal']
    edge_ys = np.unique(np.concatenate((hor_lines['y0'].values, hor_lines['y1'].values)))
    return np.sort(edge_ys)

def chars_to_words(chars, char_tol=2):
    """Convert characters to words, handling horizontal spacing"""
    words = []
    word = []
    prev_x1 = None
    for _, char in chars.iterrows():
        if prev_x1 is not None and abs(char['x0'] - prev_x1) > char_tol:
            if word:
                words.append(word)
            word = []
        word.append(char)
        prev_x1 = char['x1']
    if word:
        words.append(word)
    
    words_out = []
    for word_chars in words:
        if not word_chars:
            continue
        text = "".join([c["text"] for c in word_chars]).strip()
        x0 = word_chars[0]["x0"]
        y0 = min(c["y0"] for c in word_chars)
        x1 = word_chars[-1]["x1"]
        y1 = max(c["y1"] for c in word_chars)
        words_out.append({"text": text, "x0": x0, "y0": y0, "x1": x1, "y1": y1})
    return words_out

def check_y_overlap(word1, word2, overlap_threshold=0):
    """Check if two words have overlapping Y coordinates"""
    y1_top, y1_bottom = word1['y0'], word1['y1']
    y2_top, y2_bottom = word2['y0'], word2['y1']
    
    overlap_start = max(y1_top, y2_top)
    overlap_end = min(y1_bottom, y2_bottom)
    overlap = max(0, overlap_end - overlap_start)
    
    min_height = min(y1_bottom - y1_top, y2_bottom - y2_top)
    
    return overlap / min_height >= overlap_threshold if min_height > 0 else False

def group_words_by_logical_cells(words, x_gap_threshold=20, y_proximity_threshold=3):
    """
    Group words into logical cells by:
    1. First identifying words that are on same horizontal line (Y overlap)
    2. Then grouping vertically close text within same column area
    """
    if not words:
        return []
    
    sorted_words = sorted(words, key=lambda w: w['x0'])
    horizontal_groups = []
    
    for word in sorted_words:
        placed = False
        for group in horizontal_groups:
            if any(check_y_overlap(word, existing_word) for existing_word in group):
                group.append(word)
                placed = True
                break
        if not placed:
            horizontal_groups.append([word])
    
    logical_cells = []
    for h_group in horizontal_groups:
        if len(h_group) <= 1:
            logical_cells.extend(h_group)
            continue
            
        h_group_sorted = sorted(h_group, key=lambda w: w['x0'])
        
        columns = []
        current_column = [h_group_sorted[0]]
        
        for i in range(1, len(h_group_sorted)):
            prev_word = h_group_sorted[i-1]
            curr_word = h_group_sorted[i]
            gap = curr_word['x0'] - prev_word['x1']
            
            if gap > x_gap_threshold:
                columns.append(current_column)
                current_column = [curr_word]
            else:
                current_column.append(curr_word)
        
        if current_column:
            columns.append(current_column)
        
        for column in columns:
            if len(column) <= 1:
                logical_cells.extend(column)
            else:
                merged_cell = merge_word_group(column)
                logical_cells.append(merged_cell)
    
    x_grouped = {}
    for cell in logical_cells:
        x_key = round(cell['x0'] / 10) * 10
        if x_key not in x_grouped:
            x_grouped[x_key] = []
        x_grouped[x_key].append(cell)
    
    final_cells = []
    for x_key in sorted(x_grouped.keys()):
        # sort top-to-bottom
        column_cells = sorted(x_grouped[x_key], key=lambda c: -c['y0'])
        
        merged_column = []
        current_group = [column_cells[0]]
        
        for i in range(1, len(column_cells)):
            prev_cell = current_group[-1]
            curr_cell = column_cells[i]
            vertical_gap = curr_cell['y0'] - prev_cell['y1']
            
            if vertical_gap <= y_proximity_threshold:
                current_group.append(curr_cell)
            else:
                if len(current_group) > 1:
                    merged_cell = merge_word_group(current_group)
                    merged_column.append(merged_cell)
                else:
                    merged_column.extend(current_group)
                current_group = [curr_cell]
        
        if len(current_group) > 1:
            merged_cell = merge_word_group(current_group)
            merged_column.append(merged_cell)
        else:
            merged_column.extend(current_group)
        
        final_cells.extend(merged_column)
    
    return final_cells

def merge_word_group(word_group):
    """Merge a group of words into a single word object"""
    if len(word_group) == 1:
        return word_group[0]
    
    sorted_group = sorted(word_group, key=lambda w: (-w['y0'], w['x0']))
    combined_text = " ".join(word['text'] for word in sorted_group)
    
    x0 = min(word['x0'] for word in sorted_group)
    y0 = min(word['y0'] for word in sorted_group)
    x1 = max(word['x1'] for word in sorted_group)
    y1 = max(word['y1'] for word in sorted_group)
    
    return {
        "text": combined_text,
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1
    }

def words_to_phrases(words, phrase_tol=8):
    """Convert words to phrases with multi-line header handling"""
    if not words:
        return []
    
    logical_cells = group_words_by_logical_cells(words, x_gap_threshold=6, y_proximity_threshold=5)
    logical_cells.sort(key=lambda w: w['x0'])
    
    phrases = []
    phrase = []
    prev_x1 = None
    
    for cell in logical_cells:
        if prev_x1 is not None and abs(cell["x0"] - prev_x1) > phrase_tol:
            if phrase:
                phrase_text = " ".join(w["text"] for w in phrase).strip()
                phrases.append({
                    "text": phrase_text, 
                    "x0": phrase[0]["x0"], 
                    "y0": min(w["y0"] for w in phrase),
                    "x1": phrase[-1]["x1"], 
                    "y1": max(w["y1"] for w in phrase)
                })
            phrase = []
        phrase.append(cell)
        prev_x1 = cell["x1"]
    
    if phrase:
        phrase_text = " ".join(w["text"] for w in phrase).strip()
        phrases.append({
            "text": phrase_text, 
            "x0": phrase[0]["x0"], 
            "y0": min(w["y0"] for w in phrase),
            "x1": phrase[-1]["x1"], 
            "y1": max(w["y1"] for w in phrase)
        })
    
    return phrases

def compute_semantic_header_score(phrases):
    """Compute semantic similarity score for header detection"""
    texts = [p["text"].strip() for p in phrases if p["text"].strip()]
    if texts and all(t.upper() in COMMON_HEADERS for t in texts):
        return 0.0
    if not texts:
        return 0.0
    block_embeddings = transformer_model.encode(texts, convert_to_tensor=True)
    scores = []
    for emb in block_embeddings:
        similarities = util.pytorch_cos_sim(emb, header_keyword_embeddings)[0]
        max_sim, _ = torch.max(similarities, 0)
        scores.append(max_sim.item())
    return np.mean(scores) if scores else 0.0

def extract_header_blocks_between_edges(pdf_path, page_num=0, min_phrases_in_row=2):
    """Extract header blocks between horizontal edges with confidence scores"""
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        chars = pd.DataFrame(page.chars)
        if chars.empty:
            return []

        hor_edges = extract_horizontal_edges(page)
        if len(hor_edges) < 2:
            return []

        header_blocks = []
        for k in range(len(hor_edges) - 1):
            top, bottom = hor_edges[k], hor_edges[k + 1]
            # ensure top-to-bottom ordering for me 
            chars_band = chars[
                (chars["y0"] >= top) & (chars["y1"] <= bottom)
            ].sort_values(by=["y0", "x0"], ascending=[False, True])
            
            if chars_band.empty:
                continue
            
            words = chars_to_words(chars_band)
            phrases = words_to_phrases(words,phrase_tol=1)
            
            if len(phrases) < min_phrases_in_row:
                continue
            
            confidence = compute_semantic_header_score(phrases)
            header_blocks.append((phrases, confidence, (top, bottom)))

        header_blocks.sort(key=lambda x: x[1], reverse=True)
        return header_blocks

if __name__ == "__main__":
    pdf_folder = "../../tests/Priority Banks/Unbordered/"
    output_folder = "./header_detection_results"
    os.makedirs(output_folder, exist_ok=True)

    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        try:
            header_blocks_with_conf = extract_header_blocks_between_edges(pdf_path, page_num=0)
            top_5 = header_blocks_with_conf[:5]
            output_text = f"PDF: {pdf_file}\nTop 5 detected header blocks with confidence scores:\n"
            print(f"Processing {pdf_file}...")
            print(output_text)

            for i, (block_phrases, confidence, (top, bottom)) in enumerate(top_5):
                output_text += f"Block {i} vertical bounds ({top:.2f}, {bottom:.2f}) confidence: {confidence:.4f}\n"
                print(f"Block {i} vertical bounds ({top:.2f}, {bottom:.2f}) confidence: {confidence:.4f}")
                phrase_strs = [p["text"] for p in block_phrases if p["text"].strip()]
                output_text += "  Header phrases: " + ", ".join(phrase_strs) + "\n"
                print("  Header phrases:", ", ".join(phrase_strs))
            output_text += "\n"

            output_file_path = os.path.join(output_folder, f"{pdf_file.replace(' ', '_').replace('.pdf', '')}_headers.txt")
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(output_text)

            print(f"Results saved to {output_file_path}\n")
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}\n")
