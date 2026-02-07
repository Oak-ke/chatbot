import json
import re
import os
import logging
import pandas as pd
from langgraph.graph import StateGraph
from typing import TypedDict
from langchain_core.messages import HumanMessage
from utils import detect_language, translate_text
from llm import llama_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from visualizer import Visualizer, FileDataSource, MockDataSource
import uuid
from dotenv import load_dotenv

import matplotlib
matplotlib.use("agg") # This for headless plot graphs(Use before pyplot import)

# Configure logging
logger = logging.getLogger(__name__)
load_dotenv()
llm = llama_llm()

# State definition
class State(TypedDict):
    question: str
    language: str
    intent: str
    data: str | None
    answer: str
    graph_url: str

# Intent mapping
INTENT_MAP = {
    "system_name": ["system name", "name of system", "what is the system called"],
    "system_info": ["system_info", "system information", "about system", "info", "tell me more"],
    "cooperatives_total": ["cooperatives_total", "number of cooperatives", "cooperatives registered"],
    "members_total": ["members_total", "total members", "members", "female members", "male members"],
    "members_by_state": ["members_by_state", "members per state", "members by state"],
    "female_members": ["female members"],
    "male_members": ["male members"],
    "directors_total": ["directors_total", "number of directors", "directors"],
    "visualize": ["visualize", "graph", "plot", "chart", "show me trends"]
}

# Database setup
db_uri = os.getenv("DB_URI")
db = SQLDatabase.from_uri(db_uri)

# Allowed tables that the LLM is allowed to query
ALLOWED_TABLES = {"member", "cooperative", "director", "cooperative_location", "cooperative_stages"}

# Whitelist of allowed columns per table
ALLOWED_COLUMNS = {
    "cooperative": {"cooperative_id", "cooperative_name", "cooperative_type", "cooperative_state", "cooperative_constitution", "cooperative_bylaws", "has_directors", "cooperative_state", "cooperative_boma", "approval_statusregisted", "cooperative_certificate"},
    "member": {"cooperative_id", "member_id", "member_name", "member_gender", "member_state", "member_county", "member_payam", "member_boma"},
    "director": {"cooperative_id", "director_id", "director_name", "director_gender", "director_payam", "director_state", "director_county", "director_boma"},
    "cooperative_location": {"cooperative_id", "state", "county", "payam", "boma"},
    "cooperative_stages": {"coop_id", "stage", "status", "nexr_stage"}
}

# Helper functions
def get_schema(_):
    """
    Return schema information for allowed tables.
    Used for LLM to understand table structure when generating SQL.
    """
    return db.get_table_info(table_names=ALLOWED_TABLES)


def sanitize_sql(sql: str) -> str:
    """
    Remove any markdown code blocks and trailing semicolons from SQL.
    """
    sql = re.sub(r"```sql|```", "", sql, flags=re.IGNORECASE)
    return sql.strip().rstrip(";")

def validate_sql(sql: str) -> None:
    """
    Ensure that only allowed tables and columns are referenced in SQL.
    Raise an error if invalid tables/columns are used or no table is referenced.
    """
    sql_lower = sql.lower()
    
    # Extract table names from FROM and JOIN clauses
    tables = set(re.findall(r"\bfrom\s+(\w+)|\bjoin\s+(\w+)", sql_lower))
    tables = {t for pair in tables for t in pair if t}

    if not tables:
        raise ValueError("No table referenced in query")

    illegal_tables = tables - ALLOWED_TABLES
    if illegal_tables:
        raise ValueError(f"Illegal tables used: {illegal_tables}. Allowed: {ALLOWED_TABLES}")
    
    # Extract column names (basic extraction)
    columns = set(re.findall(r"\b(\w+)\s*(?:=|<|>|!=|\blike\b)", sql_lower))
    columns.update(re.findall(r"\bselect\s+([^,\s]+)", sql_lower))
    columns.update(re.findall(r",\s*(\w+)", sql_lower))
    
    # Remove SQL keywords and functions
    keywords = {"count", "sum", "avg", "max", "min", "distinct", "as", "and", "or", "not", "in", "like", "between"}
    columns = {c for c in columns if c and c not in keywords and not c.isdigit()}
    
    # Validate columns
    for table in tables:
        allowed = ALLOWED_COLUMNS.get(table, set())
        # Extract columns that appear to be from this table
        table_cols = set(re.findall(rf"{table}\.(\w+)", sql_lower))
        
        if table_cols:
            illegal_cols = table_cols - allowed
            if illegal_cols:
                raise ValueError(
                    f"Invalid columns for table '{table}': {illegal_cols}. "
                    f"Allowed: {allowed}"
                )
    
def run_query(query: str):
    """
    Sanitize and execute the SQL query against the database.
    Returns the query results.
    Raises descriptive errors if execution fails.
    """
    sql = sanitize_sql(query)
    try:
        result = db.run(sql)
        return result
    except Exception as e:
        error_msg = str(e)
        if "Unknown column" in error_msg:
            raise ValueError(f"Invalid column name in query. {error_msg}")
        elif "doesn't exist" in error_msg:
            raise ValueError(f"Table doesn't exist. {error_msg}")
        else:
            raise RuntimeError(f"Query execution failed: {error_msg}")

# SQL generation
def write_sql_query(llm):
    """
    Generates a Runnable that will create SQL for a user question.
    Uses a ChatPromptTemplate to instruct the LLM on exact columns to use.
    """
    sql_template = """
        You are a MySQL SQL generator for a cooperative database.

        CRITICAL: ONLY 5 TABLES EXIST IN THIS DATABASE
        The ONLY tables available are:
        1. cooperative
        2. member
        3. director
        4. cooperative_stages
        5. cooperative_location
        
        Do NOT use any other tables (reserve, person, staff, accounts, employees, etc. DO NOT EXIST).

        COMPLETE COLUMN REFERENCE (use EXACTLY ALLOWED_COLUMNS):
        
        cooperative table:
        - cooperative_id, cooperative_name, cooperative_type, cooperative_state
        - cooperative_constitution, cooperative_bylaws, has_directors, has_members
        - cooperative_county, cooperative_payam, cooperative_boma, approval_status
        - cooperative_certificate, enumerator_id, cooperative_date_created
        
        member table:
        - member_id, cooperative_id, member_name, member_gender
        - member_state, member_county, member_payam, member_boma
        
        director table:
        - director_id, cooperative_id, director_name, director_gender
        - director_state, director_county, director_payam, director_boma
        
        CRITICAL RULES:
        1. Use table.column format (e.g., member.member_gender, NOT member.gender)
        2. Do NOT invent column names or table names
        3. ALWAYS PREFER JOINs over subqueries
        4. When filtering by cooperative_name, ALWAYS use INNER JOIN: 
           SELECT ... FROM member m INNER JOIN cooperative c ON m.cooperative_id = c.cooperative_id WHERE c.cooperative_name = '...'
        5. If you must use a subquery with multiple matches, use IN not =:
           WHERE cooperative_id IN (SELECT cooperative_id FROM cooperative WHERE ...)
        6. For aggregation queries (state, count, max, etc.), query the appropriate table directly
        7. Remember: state information exists in THREE tables as different columns:
           - cooperative.cooperative_state (for cooperatives)
           - member.member_state (for members)
           - director.director_state (for directors)
        8. For location/state matching, use LOWER() for case-insensitive comparison
        9. Always include COUNT in aggregation SELECT - never just group without counting
        10. Return ONLY ONE SELECT statement, no markdown, no explanation
        
        CRITICAL FOR LOCATION QUERIES:
        When the question asks about a state, county, payam, boma or any location name:
        - WRONG: WHERE cooperative_state = 'Western Bahr el Ghazal'
        - WRONG: WHERE LOWER(cooperative_state) = 'western bahr el ghazal'
        - CORRECT: WHERE LOWER(cooperative_state) = LOWER('Western Bahr el Ghazal')
        The database stores names in mixed case, so you MUST use LOWER() on BOTH sides!
        
        EXAMPLES:
        - "female members" → SELECT COUNT(*) FROM member WHERE member_gender = 'Female'
        - "members in Yambio Farmers Cooperative" → SELECT COUNT(*) FROM member m INNER JOIN cooperative c ON m.cooperative_id = c.cooperative_id WHERE c.cooperative_name = 'Yambio Farmers Cooperative'
        - "which state has the most cooperatives" → SELECT c.cooperative_state, COUNT(*) AS count FROM cooperative c GROUP BY c.cooperative_state ORDER BY count DESC LIMIT 1
        - "how many cooperatives in Western Bahr el Ghazal" → SELECT COUNT(*) FROM cooperative WHERE LOWER(cooperative_state) = LOWER('Western Bahr el Ghazal')
        - "directors in each cooperative" → SELECT c.cooperative_name, COUNT(d.director_id) AS count FROM cooperative c LEFT JOIN director d ON c.cooperative_id = d.cooperative_id GROUP BY c.cooperative_id
        
        Database Schema:
        {schema}

        User Question:
        {question}

        Output SQL (no markdown, no explanation):
    """

    # Define LLM prompt msg
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert SQL generator for a MySQL cooperative database.

                    CRITICAL: This database has ONLY 5 TABLES:
                    1. cooperative
                    2. member  
                    3. director
                    4. cooperative_stages
                    5. cooperative_location

                    DO NOT USE any other tables: users, person, staff, accounts, employees, etc.
                                                        
                    COMPLETE VALID COLUMNS:
                    cooperative: cooperative_id, cooperative_name, cooperative_type, cooperative_state, cooperative_constitution, cooperative_bylaws, has_directors, has_members, cooperative_county, cooperative_payam, cooperative_boma, approval_status, cooperative_certificate, enumerator_id, cooperative_date_created

                    member: member_id, cooperative_id, member_name, member_gender, member_state, member_county, member_payam, member_boma

                    director: director_id, cooperative_id, director_name, director_gender, director_state, director_county, director_payam, director_boma

                    CRITICAL SQL RULES:
                    1. ALWAYS PREFER JOINs over subqueries
                    2. When filtering by cooperative_name, ALWAYS use INNER JOIN:
                    SELECT ... FROM member m INNER JOIN cooperative c ON m.cooperative_id = c.cooperative_id WHERE c.cooperative_name = '...'
                    3. If you MUST use a subquery, use IN not = when there might be multiple matches:
                    WHERE cooperative_id IN (SELECT cooperative_id FROM cooperative WHERE ...)
                    4. For state-related queries:
                    - "state" about cooperatives = cooperative.cooperative_state
                    - "state" about members = member.member_state
                    - "state" about directors = director.director_state
                    5. For state/location matching, ALWAYS use case-insensitive comparison:
                    - Use LOWER: WHERE LOWER(cooperative_state) = LOWER('Western Bahr el Ghazal')
                    6. For aggregation queries (count by state, max/min by group):
                    - Query the primary table directly, then GROUP BY
                    - Always include COUNT(*) in the SELECT when aggregating
                    - Use proper table aliases to avoid ambiguous column errors
                    7. Always use the full column name with table prefix (member.member_gender, NOT member.gender)
                    8. Generate ONLY valid SQL, no explanations or markdown.
                """
            ),
            ("human", sql_template),
        ]
    )

    # RunnablePassthrough allows the prompt to access the schema dynamically
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def generate_valid_sql(question: str, llm, max_retries: int = 3) -> str:
    """
    Generate a valid SQL query from the user question.
    If invalid SQL is generated by the LLM, retry up to max_retries times.
    Validates table and column usage before returning SQL.
    Also validates at execution time to catch runtime errors.
    """

    error_message = None

    for attempt in range(max_retries):
        sql_raw = write_sql_query(llm).invoke(
            {
                "question": question
                if not error_message
                else f"""
                        Previous SQL was INVALID.

                        Error:
                        {error_message}

                        ⚠️ KEY FIXES BASED ON ERROR:
                        
                        IF ERROR: "Subquery returns more than 1 row"
                        → Use INNER JOIN instead of subquery
                        
                        IF ERROR: "Unknown column" or "Ambiguous column"
                        → Always use table.column format (e.g., c.cooperative_state, NOT state)
                        → Check which table has the column: 
                           - cooperative.cooperative_state (for cooperatives)
                           - member.member_state (for members)
                           - director.director_state (for directors)
                        
                        IF QUESTION ABOUT: "{question}"
                        
                        FOR "HOW MANY [THING] IN [LOCATION]" QUESTIONS:
                        - Always include the COUNT in the SELECT
                        - Use CASE-INSENSITIVE matching for state names (use LOWER or similar)
                        - Example: "How many cooperatives in Western Bahr el Ghazal?"
                          → SELECT COUNT(*) FROM cooperative WHERE LOWER(cooperative_state) = LOWER('Western Bahr el Ghazal')
                        - OR if the state name is slightly different, try fuzzy matching or list all that contain the keyword
                        
                        FOR "WHICH [LOCATION] HAS THE MOST [THINGS]" QUESTIONS:
                        - Use GROUP BY with the location column
                        - ORDER BY COUNT descending
                        - Example: "Which state has most cooperatives?"
                          → SELECT c.cooperative_state, COUNT(*) AS count FROM cooperative c GROUP BY c.cooperative_state ORDER BY count DESC LIMIT 1
                        - Make sure to include BOTH the location name AND the count in the SELECT
                        
                        RULES FOR LOCATION-BASED QUERIES:
                        1. Always use full table.column format (c.cooperative_state)
                        2. For exact matches, try LOWER() for case-insensitive comparison
                        3. For aggregations, GROUP BY the location column
                        4. Always include COUNT(*) or COUNT(id) to get the number
                        5. Never mix columns without proper JOINs
                        
                        COMPLETE VALID COLUMNS (use table.column format):
                        cooperative: c.cooperative_id, c.cooperative_name, c.cooperative_type, c.cooperative_state, 
                                     c.cooperative_county, c.cooperative_payam, c.cooperative_boma
                        
                        member: m.member_id, m.cooperative_id, m.member_name, m.member_gender, m.member_state, 
                                m.member_county, m.member_payam, m.member_boma
                        
                        director: d.director_id, d.cooperative_id, d.director_name, d.director_gender, d.director_state,
                                  d.director_county, d.director_payam, d.director_boma

                        Original question:
                        {question}
                    """
            }
        )

        sql = sanitize_sql(sql_raw)
        
        try:
            # Validate syntax/structure
            validate_sql(sql)
            
            # Try executing to catch runtime column errors
            run_query(sql)
            
            return sql
        except Exception as e:
            error_message = str(e)
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {error_message}")
            
            if attempt == max_retries - 1:
                raise RuntimeError(f"Unable to generate valid SQL after {max_retries} attempts: {error_message}")

# Natural answer generation
def answer_user_query(question: str) -> str:
    """
    Executes the generated SQL query and converts the result to a simple, direct answer.
    Provides intelligent, context-aware responses based on actual data state.
    Never shows SQL details, technical information, or explanations.
    """
    try:
        sql = generate_valid_sql(question, llm)
        response = run_query(sql)
    except Exception as e:
        logger.error(f"Query generation/execution failed: {str(e)}")
        return "I'm unable to answer that question at this time."

    # Check for empty results - handle intelligently
    if not response or response.strip() == "" or response.strip() == "0 rows in set":
        # Let LLM generate context-aware response for empty results
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are answering a user's question when the database returned no results.

                        RULES FOR EMPTY/NULL RESULTS:
                        1. Do NOT say "No data found" - be more specific
                        2. Infer from the question why there might be no results
                        3. Explain the situation naturally (e.g., "No members have gender information recorded" or "This cooperative has no members yet")
                        4. Be empathetic and informative
                        5. Keep answer to 1-2 sentences
                        6. Do NOT mention SQL, queries, or technical details

                        Examples:
                        - Q: "What gender are the members?" with no results → "No members have gender information recorded in the system."
                        - Q: "How many directors does X cooperative have?" with no results → "This cooperative has no directors registered yet."
                        - Q: "Show all members in state Y" with no results → "There are no members registered in that state."
                    """
                ),
                (
                    "human",
                    f"""User Question: {question}

                    Database returned no results. Generate a natural, context-specific explanation (1-2 sentences):"""
                ),
            ]
        )
        
        messages = prompt.format_messages()
        return llm.invoke(messages).content.strip()

    # Strict prompt for non-empty results
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are answering a user's question based on SQL query results.

                    CRITICAL RULES:
                    1. Give ONLY the answer to the user's question - be DIRECT and CONCISE
                    2. Answer in ONE sentence maximum (or 2 sentences only if absolutely necessary)
                    3. Do NOT mention SQL, queries, databases, or how you got the answer
                    4. Do NOT explain technical details or debugging information
                    5. Do NOT list data entries or show raw results
                    6. For COUNT or numeric results: ALWAYS include the number in your answer
                    7. For lookup/aggregation questions: state what was found WITH the count
                    8. If result contains NULL or missing values, explain what's missing

                    WORD LIMIT: Keep your answer under 30 words.

                    IMPORTANT EXAMPLES:
                    - Q: "Which state has most cooperatives?" Result: Western Bahr el Ghazal | 5 → "Western Bahr el Ghazal has the most cooperatives with 5."
                    - Q: "How many cooperatives in X state?" Result: 5 → "X state has 5 cooperatives."
                    - Q: "How many members?" Result: 150 → "There are 150 members."
                    - Q: "Which has most?" Result: Item | 10 → "Item has the most with 10."
                """
            ),
            (
                "human",
                f"""Question: {question}
                    Result: {response}

                    Answer (ONE sentence, under 30 words, INCLUDE ALL NUMERIC VALUES):"""
            ),
        ]
    )
    
    messages = prompt.format_messages()
    answer = llm.invoke(messages).content.strip()
    # Safety truncation since max_tokens is 256
    if len(answer) > 300:
        answer = answer[:300].rsplit('.', 1)[0] + "."
    
    return answer

# Language detection and translation
def detect_lan_and_translate(state: State, llm):
    text = state["question"]
    lang = detect_language(text)
        
    if lang == "en":
        return {
            "question": text,
            "language": "en",
        }
        
    translated, _ = translate_text(text, llm, target_lang="English")
    
    return {
        "question": translated,
        "language": "ar"
    }

# Intent detection
def detect_intent(state: State, llm):
    """
    Classify the user's question into one of the known intents.
    Returns canonical intent string.
    """
    prompt = f"Classify the intent into one of: {list(INTENT_MAP.keys())}\nQuestion: {state['question']}"
    raw_intent = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()

    # Normalize to canonical intent using substring matching
    for canonical, aliases in INTENT_MAP.items():
        if any(alias.lower() in raw_intent for alias in aliases):
            return {"intent": canonical}
    return {"intent": "unknown"}


# Data selection
def select_data(state: State):
    """
    Depending on intent, either return system info or query database for relevant data.
    """
    if state["intent"] in {"system_info", "system_name"}:
        return {"data": "This system manages cooperative data dynamically from MySQL."}

    # For other intents, generate SQL and query database
    return {
        "data": answer_user_query(state["question"])
    }


def generate_answer(state: State, llm):
    """
    Prepare the final answer. Returns data if available, otherwise a fallback message.
    """
    return {"answer": state["data"] or "No data found."}


def visualize_node(state: State):
    temp_path = None
    # Determine strategy
    if os.path.exists("data/public_data.json"):
        # We need to transform the JSON structure into a flat DF for visualization if possible
        with open("data/public_data.json", "r") as f:
            raw_data = json.load(f)
        
        # If it's the members_by_state, it's easy to visualize
        if state["intent"] == "members_by_state":
            df = pd.DataFrame(list(raw_data["members_by_state"].items()), columns=["State", "Members"])
            # Temporary file strategy for this specific DF
            temp_path = f"data/temp_{uuid.uuid4().hex}.csv"
            logger.info(f"Creating temporary CSV at {temp_path} for intent '{state['intent']}'. Data shape: {df.shape}")
            df.to_csv(temp_path, index=False)
            strategy = FileDataSource(temp_path)
        else:
            strategy = MockDataSource()
    else:
        strategy = MockDataSource()
        
    try:
        viz = Visualizer(strategy)
        intent = state.get("intent", "unknown")
        safe_intent = re.sub(r'[^a-zA-Z0-9_]', '_', intent)
        output_filename = f"static/graphs/viz_{safe_intent}.png"
        graph_path = viz.analyze_and_plot(output_path=output_filename)
        
        if not graph_path:
            return {"graph_url": None}
        
        # In a web app, we want the URL relative to static
        graph_url = f"/static/graphs/{os.path.basename(graph_path)}"
        
        return {"graph_url": graph_url}
    finally:
        if temp_path and os.path.exists(temp_path):
            logger.info(f"Deleting temporary CSV at {temp_path}")
            os.remove(temp_path)

def route_to_answer(state: State):
    if state["intent"] == "visualize" or state["intent"] == "members_by_state":
        return "visualize"
    return "answer"


def build_graph(llm):
    """
    Build a StateGraph pipeline for processing user queries:
    Steps:
    1. Translate question if needed
    2. Detect intent
    3. Fetch relevant data from DB or system info
    4. Generate visualization if applicable
    5. Generate natural language answer
    """
    graph = StateGraph(State)

    graph.add_node("translate", lambda s: detect_lan_and_translate(s, llm))
    graph.add_node("intent", lambda s: detect_intent(s, llm))
    graph.add_node("data", select_data)
    graph.add_node("visualize", visualize_node)
    graph.add_node("answer", lambda s: generate_answer(s, llm))

    graph.set_entry_point("translate")
    graph.add_edge("translate", "intent")
    graph.add_edge("intent", "data")
    
    graph.add_conditional_edges(
        "data",
        route_to_answer,
        {
            "visualize": "visualize",
            "answer": "answer"
        }
    )
    graph.add_edge("visualize", "answer")

    return graph.compile()