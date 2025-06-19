#!/usr/bin/env python3
"""
Banking Data Adapter for financial document datasets
"""

import json
import time
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import random

from pydantic import BaseModel, Field

from ...core.config import get_config
from ...core.logging import get_logger
from ...core.exceptions import ProcessingError, ValidationError


class BankingDocumentType(Enum):
    """Banking document types"""
    BANK_STATEMENT = "bank_statement"
    CHECK = "check"
    INVOICE = "invoice"
    RECEIPT = "receipt"
    LOAN_APPLICATION = "loan_application"
    CREDIT_CARD_STATEMENT = "credit_card_statement"
    WIRE_TRANSFER = "wire_transfer"
    DEPOSIT_SLIP = "deposit_slip"
    WITHDRAWAL_SLIP = "withdrawal_slip"
    MORTGAGE_DOCUMENT = "mortgage_document"


class BankingDataSource(Enum):
    """Banking data sources"""
    SYNTHETIC = "synthetic"
    ANONYMIZED_REAL = "anonymized_real"
    PUBLIC_DATASET = "public_dataset"
    MOCK_API = "mock_api"


@dataclass
class BankingTransaction:
    """Banking transaction record"""
    transaction_id: str
    date: datetime
    amount: float
    description: str
    transaction_type: str  # credit, debit
    category: str
    balance: Optional[float] = None
    reference_number: Optional[str] = None
    merchant: Optional[str] = None


@dataclass
class BankingDocument:
    """Banking document record"""
    document_id: str
    document_type: BankingDocumentType
    account_number: str  # Masked/anonymized
    customer_id: str  # Anonymized
    date_range: tuple  # (start_date, end_date)
    transactions: List[BankingTransaction] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    text_content: str = ""
    structured_data: Dict[str, Any] = field(default_factory=dict)


class BankingDataConfig(BaseModel):
    """Banking data adapter configuration"""
    
    # Data generation
    num_accounts: int = Field(100, description="Number of bank accounts to generate")
    transactions_per_account: int = Field(50, description="Transactions per account")
    date_range_days: int = Field(90, description="Date range for transactions")
    
    # Document types
    document_types: List[BankingDocumentType] = Field(
        default=[BankingDocumentType.BANK_STATEMENT, BankingDocumentType.CHECK],
        description="Document types to generate"
    )
    
    # Privacy settings
    anonymize_data: bool = Field(True, description="Anonymize sensitive data")
    mask_account_numbers: bool = Field(True, description="Mask account numbers")
    use_fake_names: bool = Field(True, description="Use fake customer names")
    
    # Realism settings
    realistic_amounts: bool = Field(True, description="Generate realistic transaction amounts")
    realistic_merchants: bool = Field(True, description="Use realistic merchant names")
    include_recurring: bool = Field(True, description="Include recurring transactions")
    
    # Output settings
    include_text_content: bool = Field(True, description="Generate text content")
    include_structured_data: bool = Field(True, description="Include structured data")
    
    # Categories
    transaction_categories: List[str] = Field(
        default=[
            "groceries", "gas", "restaurants", "utilities", "insurance",
            "entertainment", "shopping", "healthcare", "travel", "salary"
        ],
        description="Transaction categories"
    )


class BankingDataAdapter:
    """
    Banking Data Adapter for financial document datasets
    
    Generates realistic banking documents including statements, checks,
    invoices, and transaction records with proper anonymization.
    """
    
    def __init__(self, config: Optional[BankingDataConfig] = None):
        self.logger = get_logger(__name__)
        self.config = config or BankingDataConfig()
        
        # Banking data templates
        self.merchant_names = [
            "WALMART SUPERCENTER", "AMAZON.COM", "SHELL OIL", "MCDONALD'S",
            "STARBUCKS", "HOME DEPOT", "TARGET", "COSTCO", "CVS PHARMACY",
            "SAFEWAY", "EXXON MOBIL", "SUBWAY", "WALGREENS", "KROGER",
            "BEST BUY", "LOWES", "CHIPOTLE", "DUNKIN DONUTS", "TRADER JOES"
        ]
        
        self.recurring_merchants = [
            "ELECTRIC COMPANY", "WATER DEPT", "CABLE/INTERNET", "PHONE COMPANY",
            "INSURANCE CO", "MORTGAGE PAYMENT", "CAR PAYMENT", "RENT PAYMENT"
        ]
        
        self.bank_names = [
            "First National Bank", "Community Trust Bank", "Regional Credit Union",
            "Metro Bank", "Valley Bank", "Citizens Bank", "Heritage Bank"
        ]
        
        self.logger.info("Banking Data Adapter initialized")
    
    def load_data(self, source: BankingDataSource = BankingDataSource.SYNTHETIC,
                 **kwargs) -> List[BankingDocument]:
        """Load banking data from specified source"""
        start_time = time.time()
        
        try:
            if source == BankingDataSource.SYNTHETIC:
                documents = self._generate_synthetic_data()
            elif source == BankingDataSource.MOCK_API:
                documents = self._load_from_mock_api()
            elif source == BankingDataSource.PUBLIC_DATASET:
                documents = self._load_public_dataset()
            else:
                raise ValueError(f"Unsupported banking data source: {source}")
            
            loading_time = time.time() - start_time
            self.logger.info(f"Loaded {len(documents)} banking documents in {loading_time:.2f}s")
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Failed to load banking data: {e}")
            raise ProcessingError(f"Banking data loading error: {e}")
    
    def _generate_synthetic_data(self) -> List[BankingDocument]:
        """Generate synthetic banking data"""
        documents = []
        
        for account_idx in range(self.config.num_accounts):
            # Generate account info
            account_number = self._generate_account_number(account_idx)
            customer_id = f"CUST_{account_idx:06d}"
            
            # Generate documents for this account
            for doc_type in self.config.document_types:
                doc = self._generate_banking_document(account_number, customer_id, doc_type)
                documents.append(doc)
        
        return documents
    
    def _generate_banking_document(self, account_number: str, customer_id: str,
                                 doc_type: BankingDocumentType) -> BankingDocument:
        """Generate a single banking document"""
        # Generate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.date_range_days)
        
        # Generate transactions
        transactions = self._generate_transactions(start_date, end_date)
        
        # Create document
        document = BankingDocument(
            document_id=f"{doc_type.value}_{account_number}_{int(time.time())}",
            document_type=doc_type,
            account_number=account_number,
            customer_id=customer_id,
            date_range=(start_date, end_date),
            transactions=transactions,
            metadata={
                "bank_name": random.choice(self.bank_names),
                "statement_period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                "account_type": random.choice(["checking", "savings", "credit"]),
                "currency": "USD"
            }
        )
        
        # Generate content based on document type
        if self.config.include_text_content:
            document.text_content = self._generate_text_content(document)
        
        if self.config.include_structured_data:
            document.structured_data = self._generate_structured_data(document)
        
        return document
    
    def _generate_account_number(self, account_idx: int) -> str:
        """Generate account number (masked if configured)"""
        full_number = f"{random.randint(1000, 9999)}{account_idx:08d}"
        
        if self.config.mask_account_numbers:
            return f"****{full_number[-4:]}"
        else:
            return full_number
    
    def _generate_transactions(self, start_date: datetime, end_date: datetime) -> List[BankingTransaction]:
        """Generate transaction list"""
        transactions = []
        current_balance = random.uniform(1000, 10000)
        
        # Generate regular transactions
        for i in range(self.config.transactions_per_account):
            # Random date within range
            days_diff = (end_date - start_date).days
            random_days = random.randint(0, days_diff)
            transaction_date = start_date + timedelta(days=random_days)
            
            # Transaction details
            is_credit = random.random() < 0.3  # 30% chance of credit
            
            if is_credit:
                amount = random.uniform(500, 3000)  # Credits are larger
                merchant = "DIRECT DEPOSIT" if random.random() < 0.7 else "TRANSFER IN"
                category = "salary" if "DEPOSIT" in merchant else "transfer"
            else:
                amount = -random.uniform(5, 200)  # Debits are smaller
                merchant = random.choice(self.merchant_names)
                category = random.choice(self.config.transaction_categories)
            
            current_balance += amount
            
            transaction = BankingTransaction(
                transaction_id=f"TXN_{i:06d}",
                date=transaction_date,
                amount=amount,
                description=f"{merchant} PURCHASE",
                transaction_type="credit" if is_credit else "debit",
                category=category,
                balance=current_balance,
                reference_number=f"REF{random.randint(100000, 999999)}",
                merchant=merchant
            )
            
            transactions.append(transaction)
        
        # Add recurring transactions if enabled
        if self.config.include_recurring:
            recurring = self._generate_recurring_transactions(start_date, end_date, current_balance)
            transactions.extend(recurring)
        
        # Sort by date
        transactions.sort(key=lambda t: t.date)
        
        # Recalculate balances
        running_balance = random.uniform(1000, 5000)
        for transaction in transactions:
            running_balance += transaction.amount
            transaction.balance = running_balance
        
        return transactions
    
    def _generate_recurring_transactions(self, start_date: datetime, end_date: datetime,
                                       base_balance: float) -> List[BankingTransaction]:
        """Generate recurring transactions (utilities, rent, etc.)"""
        recurring = []
        
        # Monthly recurring transactions
        current_date = start_date
        transaction_id = 90000
        
        while current_date <= end_date:
            for merchant in self.recurring_merchants:
                if random.random() < 0.8:  # 80% chance each month
                    amount = {
                        "ELECTRIC COMPANY": random.uniform(80, 200),
                        "WATER DEPT": random.uniform(30, 80),
                        "CABLE/INTERNET": random.uniform(60, 120),
                        "PHONE COMPANY": random.uniform(40, 100),
                        "INSURANCE CO": random.uniform(150, 300),
                        "MORTGAGE PAYMENT": random.uniform(1200, 2500),
                        "CAR PAYMENT": random.uniform(200, 600),
                        "RENT PAYMENT": random.uniform(800, 2000)
                    }.get(merchant, random.uniform(50, 200))
                    
                    transaction = BankingTransaction(
                        transaction_id=f"REC_{transaction_id:06d}",
                        date=current_date,
                        amount=-amount,
                        description=f"{merchant} AUTOPAY",
                        transaction_type="debit",
                        category="utilities" if "COMPANY" in merchant or "DEPT" in merchant else "housing",
                        merchant=merchant
                    )
                    
                    recurring.append(transaction)
                    transaction_id += 1
            
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        
        return recurring
    
    def _generate_text_content(self, document: BankingDocument) -> str:
        """Generate text content for document"""
        if document.document_type == BankingDocumentType.BANK_STATEMENT:
            return self._generate_statement_text(document)
        elif document.document_type == BankingDocumentType.CHECK:
            return self._generate_check_text(document)
        else:
            return f"{document.document_type.value.replace('_', ' ').title()} for account {document.account_number}"
    
    def _generate_statement_text(self, document: BankingDocument) -> str:
        """Generate bank statement text"""
        bank_name = document.metadata.get("bank_name", "Bank")
        start_date, end_date = document.date_range
        
        text = f"""{bank_name.upper()}
ACCOUNT STATEMENT

Account Number: {document.account_number}
Customer ID: {document.customer_id}
Statement Period: {start_date.strftime('%m/%d/%Y')} - {end_date.strftime('%m/%d/%Y')}

TRANSACTION HISTORY:\n"""
        
        for txn in document.transactions[:10]:  # Show first 10 transactions
            text += f"{txn.date.strftime('%m/%d')} {txn.description:<30} ${txn.amount:>8.2f} ${txn.balance:>10.2f}\n"
        
        text += f"\nTotal Transactions: {len(document.transactions)}\n"
        text += f"Ending Balance: ${document.transactions[-1].balance:.2f}\n" if document.transactions else ""
        
        return text
    
    def _generate_check_text(self, document: BankingDocument) -> str:
        """Generate check text"""
        bank_name = document.metadata.get("bank_name", "Bank")
        
        if document.transactions:
            txn = document.transactions[0]
            amount = abs(txn.amount)
            
            text = f"""{bank_name.upper()}
CHECK #{random.randint(1000, 9999)}

Pay to the Order of: {txn.merchant}
Amount: ${amount:.2f}
Memo: {txn.description}
Date: {txn.date.strftime('%m/%d/%Y')}
Account: {document.account_number}"""
        else:
            text = f"{bank_name} Check for account {document.account_number}"
        
        return text
    
    def _generate_structured_data(self, document: BankingDocument) -> Dict[str, Any]:
        """Generate structured data"""
        return {
            "account_info": {
                "account_number": document.account_number,
                "customer_id": document.customer_id,
                "account_type": document.metadata.get("account_type"),
                "bank_name": document.metadata.get("bank_name")
            },
            "statement_info": {
                "start_date": document.date_range[0].isoformat(),
                "end_date": document.date_range[1].isoformat(),
                "num_transactions": len(document.transactions),
                "total_credits": sum(t.amount for t in document.transactions if t.amount > 0),
                "total_debits": sum(t.amount for t in document.transactions if t.amount < 0)
            },
            "transactions": [
                {
                    "id": txn.transaction_id,
                    "date": txn.date.isoformat(),
                    "amount": txn.amount,
                    "type": txn.transaction_type,
                    "description": txn.description,
                    "category": txn.category,
                    "merchant": txn.merchant
                }
                for txn in document.transactions[:20]  # First 20 transactions
            ]
        }
    
    def _load_from_mock_api(self) -> List[BankingDocument]:
        """Load data from mock API (placeholder)"""
        # Simulate API call
        time.sleep(0.1)
        
        # Return synthetic data for now
        return self._generate_synthetic_data()
    
    def _load_public_dataset(self) -> List[BankingDocument]:
        """Load from public banking dataset (placeholder)"""
        # In practice, load from actual public datasets
        return self._generate_synthetic_data()
    
    def get_adapter_info(self) -> Dict[str, Any]:
        """Get adapter information"""
        return {
            "adapter_type": "banking_data",
            "supported_sources": [source.value for source in BankingDataSource],
            "document_types": [doc_type.value for doc_type in self.config.document_types],
            "num_accounts": self.config.num_accounts,
            "transactions_per_account": self.config.transactions_per_account,
            "anonymized": self.config.anonymize_data
        }


# Factory function
def create_banking_adapter(**config_kwargs) -> BankingDataAdapter:
    """Factory function to create banking data adapter"""
    config = BankingDataConfig(**config_kwargs)
    return BankingDataAdapter(config)