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

# Standard header variations for mapping
HEADER_VOCAB_VARIATIONS_1 = {
    #Mandatory field

    "sr_no": ["sr no", "srno", "sr.no.", "serial", "s.no", "serial number", "no.","srl", "sl. no.", "#"],

    "date": ["date", "transaction date", "Trans Date", "Trans Dt", "TRN DT", "Txn date", "date of transaction", "payment date", "entry date", "Tran Date", "TRN. Date", 
             "completion time","date id", "post date"],

    "value_date": ["value date", "val date", "value dt", "value"],

    "description": ["description", "desc", "particulars", "details", "narration", "narrative", "transaction details", "remarks", "transaction description", "details of transaction"],
    "amount": ["amount", "amt", "value", "transaction amount"],

    "transaction_type": ["transaction type", "txn type", "type", "credits/debits", "cr/dr", "credit/debit", "credit or debit","type", "dr / cr","Withdrawal (Dr)/Deposit (Cr)","DEBIT/CREDIT"],

    "debit": ["debit", "withdrawal", "payment", "dr", "dr amount", "dr.","withdrawl(dr)","withdrawals","debits","withdrawal amt.","transaction debit amount","debit amt", "paid in","withdraw(dr amount)"],
    
    "credit": ["credit", "deposit", "receipt", "cr", "cr amount","cr.","deposit(cr)","deposits","credits", "deposit amt.","transaction credit amount","credit amt", "withdrawn","deposit(cr amount)"],
    
    #Non Mandatory field
    "beneficiary_name": ["beneficiary", "payee", "dealer name", "name", "party", "recipient"],
    "reference_number": ["ref", "ref no", "reference", "customer ref", "cust ref no", "transaction id", "utr", "cheque/ref", "chq/ref no", "cheque number", "chq no","chq/ref",
                         "chq.", "ref.no", "chq.no.","chq-no","chq/ref.no", "Cheque No.", "chq./req. number","Cheque/Ref.No.","chq./ref.no.", "chequeno.","chq. no.", "chq no/ref no","chq./ref. number",
                         "ref. no","chq / ref number", "cheque/Reference#", "chq / ref no.", "cheque#", "receipt no","chq. no.","ref no./cheque no.","cheque no/ reference no","cheque no/reference no","ref num",
                         "utr number", "utr","transaction id","tran id","instrument no", "instrument number", "inst no", "inst number", "instr. no.","instr no",
                      "Instruments", "instrmnt number","instrument id"],
    "balance": ["balance", "bal", "closing balance", "running balance", "available balance", "available bal.","closing bal","Total Amount Dr/Cr","total amount","balance amt"],
    "none": ["none"] ,
}