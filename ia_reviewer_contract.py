from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from agno.agent import Agent
from agno.team import Team
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.vectordb.lancedb import LanceDb
from agno.media import File
from agno.models.openai import OpenAIChat
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.embedder.openai import OpenAIEmbedder
from pypdf import PdfReader
from agno.vectordb.search import SearchType
from rich.pretty import pprint

user_id="user_1"
session_id="session_4"

class RiskLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class ClauseAnalysis(BaseModel):
    clause_title: str = Field(description="Title or label of the clause")
    category: Optional[str] = Field(description="Flexible category name or keyword (free-text)")
    original_text: str = Field(description="the origonal text from the contract")
    risk_level: RiskLevel
    rationale: Optional[str] = Field(description="Explanation of why this clause is flagged")
    redline_suggestion: Optional[str] = Field(description="Suggested revision or improvement")
    playbook_match: Optional[bool] = Field(description="Does this clause align with internal guidelines?")
    requires_user_input: Optional[bool] = Field(default=False, description="Does this clause need clarification from user?")
    user_attention_score: Optional[float] = Field(ge=0.0, le=1.0, description="How much the user should pay attention to this clause (0-1)")
    agent_confidence_score: Optional[float] = Field(ge=0.0, le=1.0, description="AI's confidence in this analysis")

class RiskItem(BaseModel):
    clause_title: str
    category: Optional[str]
    risk_level: RiskLevel
    reason: Optional[str]
    user_attention_score: Optional[float]
    agent_confidence_score: Optional[float]

class ReviewSummary(BaseModel):
    overall_compliance_status: str = Field(description="Final decision: Compliant / Partial / Risky")
    executive_summary: str = Field(description="High-level summary of issues")
    total_clauses_reviewed: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int

class ContractReviewBaseModel(BaseModel):
    contract_type: str = Field(description="Type of the contract (e.g., NDA, SaaS)")
    jurisdiction: Optional[str] = Field(description="Applicable governing law or location")
    clause_analyses: List[ClauseAnalysis] = Field(description="Detailed analysis of clauses")
    summary: ReviewSummary
    recommendations: List[str] = Field(description="Recommended next actions or edits")
    questions_for_user: Optional[List[str]] = Field(description="Clarification questions to refine review")

class UserContext(BaseModel):
    full_name: Optional[str] = None
    job_title: Optional[str] = None
    domain_industry: Optional[str] = None
    nationality: Optional[str] = None
    country: Optional[str] = None
    company_size: Optional[str] = None
    year_experience: Optional[str] = None
    specific_concerns: Optional[str] = None
    additional_context: Optional[str] = None

contract_knowledge = PDFKnowledgeBase(
    # path=[
    #     {
    #     "path": Path("united_educators_checklist_guide_for_reviewing_contracts.pdf"),
    #     "metadata": {"document_type": "contract_checklist", "source": "united_educators"}
    #     }
    #     ]
    path=Path("united_educators_checklist_guide_for_reviewing_contracts.pdf"),
    vector_db=LanceDb(
        uri="tmp/lancedb",
        search_type= SearchType.vector,
        table_name="contract_review_knowledge",
        embedder=OpenAIEmbedder(
            id="text-embedding-3-small", 
            api_key="youe api")
    ),

)

# read_agent = Agent(
#     name="Read file Agent",
#     role="Read file",
#     model=OpenAIChat(id="gpt-4o-mini", api_key="youe api"),
#     tools=[{"type": "file_search"}, {"type": "web_search_preview"}],
#     markdown=True,
# )

import os

# Get absolute path
current_dir = os.path.dirname(os.path.abspath(__file__))
contract_file = os.path.join(current_dir, "ServicesAgreementSample.pdf")

_cached_text = None
_cached_filename = None

def extract_text_from_pdf():
    """Extract text from a PDF file - cached version"""
    global _cached_text, _cached_filename
    
    if _cached_text is not None:
        return _cached_text
    
    reader = PdfReader(contract_file)
    full_text = "".join(page.extract_text() or "" for page in reader.pages)
    if not full_text:
        raise ValueError("No text found in the contract.")
    
    _cached_text = full_text
    _cached_filename = os.path.basename(contract_file)
    
    return full_text


def get_document() -> dict:
    """Function to get document content - properly defined as a tool function"""
    full_text = extract_text_from_pdf()
    filename = os.path.basename(contract_file)
    return {
        "content": full_text,
        "filename": filename
    }

contract_analyzer = Agent(
    name="Contract Analysis Agent",
    role="Analyze contract clauses and identify key terms",
    model=OpenAIChat(id="gpt-4o-mini", api_key="youe api"),
    tools=[get_document],
    instructions="""
    You are a contract clause analysis specialist. Your responsibility is to extract and categorize contract clauses and identify key contractual terms.

    **DOCUMENT ACCESS:**
    - Use the get_document() tool to access and retrieve the contract text that needs to be reviewed
    - The tool will provide you with the full contract content for analysis

    **FOCUS ON CLAUSE IDENTIFICATION AND EXTRACTION:**

    1. **Core Contract Terms Analysis:**
    - Identify parties and their legal status
    - Extract promises, rights, and obligations of each party
    - Locate contract duration, renewal terms, and performance milestones
    - Find modification procedures and requirements
    - Identify termination clauses and remedies for nonperformance
    - Extract dispute resolution mechanisms

    2. **Clause Categorization:**
    - Parties and entities
    - Payment and financial obligations
    - Goods, services, facilities descriptions
    - Duration and renewal terms
    - Modification procedures
    - Termination and breach remedies
    - Dispute resolution and governing law
    - Third-party liability and indemnification
    - Insurance requirements
    - Signature authority requirements

    3. **Key Terms Extraction:**
    - Extract exact clause text from the contract
    - Provide clear clause titles and categories
    - Note any referenced external documents or standards
    - Identify critical dates, deadlines, and monetary amounts
    - Flag ambiguous language that may need clarification

    **PROCESS:**
    1. Use the get_document() tool to retrieve the contract
    2. Systematically analyze each section for key terms
    3. Categorize clauses by type and importance
    4. Extract relevant text with proper context

    Focus on accurate clause identification, key terms extraction, and clear categorization of contractual elements.
    """,
    show_tool_calls=True,
    markdown=True,
)

ue_contract_reviewer = Agent(
    name="United Educators Contract Review Agent",
    role="Conduct comprehensive contract reviews following United Educators checklist methodology",
    model=OpenAIChat(id="gpt-4o-mini", api_key="youe api"),
    tools=[get_document],
    knowledge=contract_knowledge,
    search_knowledge=True,
    instructions="""
You are a contract review specialist with built-in knowledge of the United Educators checklist methodology. You have direct access to the United Educators contract review guide in your knowledge base. Use get_document() tool to access contracts for systematic analysis.

**BUILT-IN KNOWLEDGE ACCESS:**
- Your knowledge base contains the complete United Educators contract review checklist and guidelines
- Reference your knowledge base to ensure strict adherence to the established methodology
- Apply the specific criteria and standards from the United Educators guide

**UNITED EDUCATORS CHECKLIST REVIEW:**

**A. INITIAL ANALYSIS:**
- Complete read-through understanding all binding terms
- Identify ambiguous language and institutional performance capability
- Flag concerning terms for further review

**B. CORE CONTRACT TERMS:**
1. **Parties:** Verify correct identification, legal status, assignment rights
2. **Promises & Obligations:** Complete purpose description, payment amounts, goods/services details, external references
3. **Duration:** Contract dates, performance milestones, renewal provisions
4. **Modifications:** Required procedures, unilateral rights, written requirements
5. **Remedies:** Breach consequences, termination provisions, force majeure, cure opportunities
6. **Disputes:** Arbitration, mediation, governing law, venue, attorney fees

**C. THIRD-PARTY RISK:**
1. **Risk Allocation:** Identify indemnification/liability clauses, categorize as one-sided/intermediate/limited
2. **Insurance:** Document requirements, limits, certificates, additional insured provisions

**D. SIGNATURE AUTHORITY:**
- Confirm signing authority for both parties
- Verify correct names and titles

**E. GENERAL REVIEW:**
- Check spelling, formatting, grammar, professional appearance

**PROCESS:**
1. Retrieve contract via get_document()
2. Apply your built-in United Educators knowledge base guidelines
3. Complete each checklist section systematically per the guide
4. Use Yes/No/Don't Know assessments as specified in your knowledge base
5. Flag items requiring legal counsel or risk manager consultation
6. Provide risk assessment and recommendations based on established criteria

Reference your knowledge base throughout the review to ensure compliance with United Educators standards and methodology.
""",
    show_tool_calls=True,
    markdown=True,
)


risk_assessor = Agent(
    name="Risk Assessment Agent",
    role="Evaluate contract risks and compliance",
    model=OpenAIChat(id="gpt-4o-mini", api_key="youe api"),
    instructions="""
 You are a legal risk assessment specialist. Your single responsibility is to evaluate risks and assign priority levels to contract clauses.

    FOCUS ONLY ON RISK EVALUATION AND SCORING:

    1. **Risk Level Assignment (HIGH/MEDIUM/LOW/UNKNOWN):**
       - HIGH: Broad indemnification, unlimited liability, unfavorable jurisdiction, automatic renewal without notice, severe penalty clauses
       - MEDIUM: Joint liability provisions, binding arbitration, out-of-state dispute resolution, limited termination rights
       - LOW: Standard mutual provisions, reasonable insurance requirements, typical payment terms
       - UNKNOWN: Ambiguous language requiring clarification

    2. **Risk Scoring (0.0 to 1.0):**
       - user_attention_score: How much user focus this clause needs
       - agent_confidence_score: Your confidence in the risk assessment

    3. **Risk Analysis Categories:**
       - Financial exposure and liability limits
       - Legal compliance and regulatory issues
       - Operational constraints and performance risks
       - Termination and breach consequences
       - Insurance adequacy and coverage gaps
       - Jurisdiction and dispute resolution disadvantages

    4. **Compliance Assessment:**
       - Overall compliance status: Compliant/Partial/Risky
       - Specific compliance gaps or violations
       - Missing standard protective clauses

    Focus solely on risk evaluation, scoring, and compliance assessment.
    Provide clear rationale for each risk level assignment.
    """,
    tools=[get_document],
    show_tool_calls=True,
    markdown=True

)

contract_storage = SqliteAgentStorage(
        table_name="contract_sessions",
        db_file="tmp/contract_memory.db"
    )


user_context = UserContext(
    full_name="John Smith",
    nationality="Morocco",
    job_title="Software Engineering",
    domain_industry="Technology",
    company_size="startup",
    year_experience="1an",
    additional_context="First time reviewing SaaS agreements"
)


contract_review_team = Team(
    name="Contract Review Team",
    mode="collaborate",
    model=OpenAIChat(id="gpt-4o-mini", api_key="youe api"),
    members=[contract_analyzer, risk_assessor, ue_contract_reviewer],
    response_model=ContractReviewBaseModel,
    parser_model=OpenAIChat(id="gpt-4o-mini", api_key="youe api"),
    instructions="""
    You are a contract review team that provides comprehensive legal document analysis.

    You have 3 agent to collect information about contract:
    1. Contract Analysis Agent
    2. Risk Assessment Agent
    3. United Educators Contract Review Agent

    Process:
    1. call agents directly each one has the acess to the file
    2. Then, have the Contract Analysis Agent review the document structure and detailed clause analysis
    2. Then, have the Risk Assessment Agent evaluate risks with scoring and compliance assessment
    3. Have the User Needs Agent identify areas needing user clarification and generate questions
    4. Combine findings into a structured report with:
    - Detailed clause analyses with confidence scores
    - Risk items with attention scores
    - Comprehensive summary with compliance status
    - Prioritized recommendations
    - Questions for user clarification
    5. Ensure all risk levels are properly categorized as HIGH, MEDIUM, LOW, or UNKNOWN
    6. Calculate accurate counts for the summary section
    """,
    enable_agentic_context=True,
    markdown=True,
    show_members_responses=True,
    get_member_information_tool=True,
    storage=contract_storage,
    debug_mode=True,
)

# contract_knowledge.load(recreate=True)
contract_knowledge.load()

contract_review_team.print_response(
    f"Please analyze the contract. User context: {user_context.model_dump_json()}",
    user_id=user_id,
    session_id=session_id
)

pprint(contract_review_team.run_response.content)
# def review_contract(contract_file_path: str = None, user_context: Optional[UserContext] = None):
    
#     # contract_path = Path(contract_file_path)
#     contract_path = Path(contract_file_path).resolve()
    
#     if not contract_path.exists():
#         raise FileNotFoundError(f"Contract file not found: {contract_path}")
    
    
#     contract_review_team.print_response(
#         """
# Please analyze this contract, it a format pdf
#         """,
#         files=[File(url="https://community.pepperdine.edu/hr/content/forms/united_educators_checklist_guide_for_reviewing_contracts.pdf")],
#         user_id=user_id,
#         session_id=session_id,
#     )



    # review_contract("ServicesAgreementSample.pdf",user_context)
    # review_contract()

# if __name__ == "__main__":
#     import os
    
#     # Get absolute path
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     contract_file = os.path.join(current_dir, "ServicesAgreementSample.pdf")
    
#     review_contract(contract_file)



