import re
import os
import logging
import pandas as pd
from langgraph.graph import StateGraph
from typing import TypedDict, Optional
from langchain_core.messages import HumanMessage
from utils import detect_language, translate_text
from llm import llama_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import base64
from io import BytesIO
from dotenv import load_dotenv
import matplotlib
matplotlib.use("agg") # This for headless plot graphs(Use before pyplot import)
import matplotlib.pyplot as plt

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

load_dotenv()
llm = llama_llm()

# State definition
class State(TypedDict):
    question: str
    language: str
    intent: str
    data: str | None
    answer: str
    viz_data: Optional[pd.DataFrame]
    graph_base64: Optional[str]

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


def extract_tables_and_aliases(sql_text: str):
    """
    Extract table names and their aliases from SQL query.
    Returns tuple of (set of tables, dict mapping aliases to table names).
    """
    tables = set()
    alias_map = {}
    
    # Find FROM clause tables
    from_pattern = r"\bfrom\s+(\w+)(?:\s+(?:as\s+)?(\w+))?"
    for match in re.finditer(from_pattern, sql_text):
        table = match.group(1)
        alias = match.group(2) or table
        tables.add(table)
        alias_map[alias] = table
    
    # Find JOIN clause tables
    join_pattern = r"\bjoin\s+(\w+)(?:\s+(?:as\s+)?(\w+))?"
    for match in re.finditer(join_pattern, sql_text):
        table = match.group(1)
        alias = match.group(2) or table
        tables.add(table)
        alias_map[alias] = table
    
    return tables, alias_map


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
    
def log_index_usage(sql: str):
    """
    Runs EXPLAIN on the SQL query and logs whether indexes are being used.
    """
    try:
        explain_sql = f"EXPLAIN {sql}"
        explain_result = pd.read_sql(explain_sql, db._engine)
        
        # Check if any rows show 'type' not equal to 'ALL' (i.e., index used)
        explain_result['index_used'] = explain_result['type'].apply(lambda t: t != 'ALL')
        logger.info(f"[EXPLAIN RESULT]\n{explain_result[['table','type','key','rows','index_used']]}")
        
    except Exception as e:
        logger.warning(f"Failed to log index usage: {e}")
        
def run_query(query: str):
    """
    Sanitize and execute the SQL query against the database.
    Returns the query results.
    Raises descriptive errors if execution fails.
    """
    sql = sanitize_sql(query)
    logger.info(f"[SQL GENERATED] {sql}")
    log_index_usage(sql)

    try:
        result = db.run(sql)
        logger.info(f"[SQL RESULT] {str(result)[:500]}")
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
        
        Do NOT use any other tables (reserve, admin, citizen, invoices, note, password_reset, receipts. DO NOT EXIST).

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
        IMPORTANT: State names in the database use UNDERSCORES, not spaces!
        - User types: 'Western Bahr el Ghazal' (with spaces)
        - Database has: 'Western_Bahr_el_Ghazal' (with underscores)
        
        Use REPLACE to convert spaces to underscores in comparison:
        - CORRECT: WHERE LOWER(cooperative_state) = LOWER(REPLACE('Western Bahr el Ghazal', ' ', '_'))
        OR normalize the column:
        - CORRECT: WHERE LOWER(REPLACE(cooperative_state, '_', ' ')) = LOWER('Western Bahr el Ghazal')
        
        WRONG approaches:
        - WHERE cooperative_state = 'Western Bahr el Ghazal' (case AND format mismatch)
        - WHERE LOWER(cooperative_state) = LOWER('Western Bahr el Ghazal') (missing underscore conversion)
        
        EXAMPLES:
        - "female members" → SELECT COUNT(*) FROM member WHERE member_gender = 'Female'
        - "members in Yambio Farmers Cooperative" → SELECT COUNT(*) FROM member m INNER JOIN cooperative c ON m.cooperative_id = c.cooperative_id WHERE c.cooperative_name = 'Yambio Farmers Cooperative'
        - "which state has the most cooperatives" → SELECT REPLACE(c.cooperative_state, '_', ' ') AS state, COUNT(*) AS count FROM cooperative c GROUP BY c.cooperative_state ORDER BY count DESC LIMIT 1
        - "how many cooperatives in Western Bahr el Ghazal" → SELECT COUNT(*) FROM cooperative WHERE LOWER(cooperative_state) = LOWER(REPLACE('Western Bahr el Ghazal', ' ', '_'))
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

                    If you use ANY table other than the 5 listed, your output is INVALID.
                    CRITICAL: This database has ONLY 5 TABLES:
                    1. cooperative
                    2. member  
                    3. director
                    4. cooperative_stages
                    5. cooperative_location

                    DO NOT USE any other tables: reserve, citizen, admin, invoices, note, password_reset, receipts.
                                                        
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

    # autocorrect helper: replace alias.cooperative_state with the appropriate
    # member_state/director_state column when the prefix refers to the
    # member or director table.  This helps fix common LLM mistakes.
    def autocorrect_state_aliases(sql_text: str) -> str:
        tbls, alias_map = extract_tables_and_aliases(sql_text.lower())
        def repl(match):
            prefix = match.group(1)
            real = alias_map.get(prefix, prefix)
            if real == 'member':
                return f"{prefix}.member_state"
            if real == 'director':
                return f"{prefix}.director_state"
            return match.group(0)
        return re.sub(r"\b(\w+)\.cooperative_state\b", repl, sql_text)

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

                        KEY FIXES BASED ON ERROR:
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
                        - CRITICAL: State names in database use UNDERSCORES not SPACES!
                        - User input: 'Western Bahr el Ghazal' (spaces)
                        - Database: 'Western_Bahr_el_Ghazal' (underscores)
                        - CORRECT: SELECT COUNT(*) FROM cooperative WHERE LOWER(cooperative_state) = LOWER(REPLACE('Western Bahr el Ghazal', ' ', '_'))
                        - OR: SELECT COUNT(*) FROM cooperative WHERE LOWER(REPLACE(cooperative_state, '_', ' ')) = LOWER('Western Bahr el Ghazal')
                        - WRONG: WHERE LOWER(cooperative_state) = LOWER('Western Bahr el Ghazal') (spaces don't match underscores!)
                        
                        FOR "WHICH [LOCATION] HAS THE MOST [THINGS]" QUESTIONS:
                        - Use GROUP BY with the location column
                        - ORDER BY COUNT descending  
                        - CONVERT underscores to spaces for display:
                        - CORRECT: SELECT REPLACE(c.cooperative_state, '_', ' ') AS state, COUNT(*) AS count FROM cooperative c GROUP BY c.cooperative_state ORDER BY count DESC LIMIT 1
                        - This will return 'Western Bahr el Ghazal' instead of 'Western_Bahr_el_Ghazal'
                        
                        RULES FOR LOCATION-BASED QUERIES:
                        1. Always use full table.column format (c.cooperative_state)
                        2. For matching user input to database: use REPLACE to convert spaces ↔ underscores
                        3. For aggregations, GROUP BY the location column
                        4. Always include COUNT(*) or COUNT(id) to get the number
                        5. ALWAYS use REPLACE() when comparing with user-provided location names
                        
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
        # apply auto‑corrections before validation/execution
        sql = autocorrect_state_aliases(sql)
        try:
            # Validate syntax/structure
            validate_sql(sql)
            
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
        logger.info(f"[FINAL SQL USED] {sql}")
        response = run_query(sql)
    except Exception as e:
        logger.error(f"Query generation/execution failed: {str(e)}")
        return "I'm unable to answer that question at this time."

    # Check for empty results
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
        return {"answer": "Co-op Magic is a comprehensive system designed to manage cooperatives' data across South Sudan securely and efficiently. With Co-op Magic, users can easily register, track, and manage cooperative members, and operations all in one centralized platform. The system ensures data integrity and security while providing actionable insights to help cooperatives operate smoothly and transparently."}

    if state["intent"] == "visualize":
        sql = generate_valid_sql(state["question"], llm)
        df = run_query_df(sql)
        # Generate a caption for the visualization
        answer = answer_user_query(state["question"])
        return {"viz_data": df, "answer": answer}
    
    # For other intents, generate SQL and query database
    return {
        "answer": answer_user_query(state["question"])
    }


def generate_answer(state: State, llm):
    """
    Prepare the final answer. Returns data if available, otherwise a fallback message.
    """
    result = {"answer": state.get("answer") or "No data found."}
    if "graph_base64" in state and state["graph_base64"]:
        result["graph_base64"] = state["graph_base64"]
    
    return result


def run_query_df(query: str) -> pd.DataFrame:
    sql = sanitize_sql(query)
    logger.info(f"[SQL DF] {sql}")
    log_index_usage(sql)
    return pd.read_sql(sql, db._engine)


def detect_chart_type(question: str) -> str:
    """
    Detect what type of chart the user is requesting from their question.
    Returns: 'pie', 'bar', 'line', 'histogram', or 'bar' (default)
    """
    question_lower = question.lower()
    
    if "pie" in question_lower or "pie chart" in question_lower:
        return "pie"
    elif "line" in question_lower or "trend" in question_lower or "over time" in question_lower:
        return "line"
    elif "histogram" in question_lower or "distribution" in question_lower and "histogram" in question_lower:
        return "histogram"
    elif "bar" in question_lower or "bar chart" in question_lower:
        return "bar"
    else:
        return "bar"  # Default to bar chart


def visualize_node(state: State):
    """
    Generate a chart from the state's viz_data DataFrame and return it as a Base64 string.
    Supports pie, bar, line, and histogram charts.
    If viz_data is missing or empty, returns None for the graph.
    Normalizes data for cleaner visualizations.
    """
    df = state.get("viz_data")

    if not isinstance(df, pd.DataFrame) or df.empty:
        logger.warning("No dataframe available for visualization")
        return {"graph_base64": None}

    # Normalize gender values (handle inconsistency: ' M ', 'M', 'Male', etc.)
    if len(df.columns) >= 1:
        gender_col = df.columns[0]
        
        # Check if this looks like gender data
        if gender_col.lower().strip() in ["member_gender", "director_gender"]:
            def normalize_gender(val):
                val = str(val).strip().lower() if val else "unknown"
                if val in ["m", "male"]:
                    return "Male"
                elif val in ["f", "female"]:
                    return "Female"
                elif val == "unknown" or val == "":
                    return "Other"
                else:
                    return val.capitalize()
            
            df[gender_col] = df[gender_col].apply(normalize_gender)
            
            # Group by normalized values and sum counts
            if len(df.columns) >= 2:
                count_col = df.columns[1]
                df = df.groupby(gender_col)[count_col].sum().reset_index()
                logger.info(f"[VIZ DATA NORMALIZED] {df.to_dict('records')}")

    # Detect chart type from question
    chart_type = detect_chart_type(state.get("question", ""))
    logger.info(f"[CHART TYPE DETECTED] {chart_type}")

    # Create appropriate chart
    plt.figure(figsize=(10, 6))
    
    if chart_type == "pie":
        # Pie chart with better colors and larger fonts
        colors = ['#FF6B6B', '#4ECDC4', "#BCD145", '#FFA07A', '#98D8C8', '#F7DC6F']
        _, texts, autotexts = plt.pie(
            df[df.columns[1]], 
            labels=df[df.columns[0]], 
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 13, 'weight': 'bold'}
        )
        plt.title(f'{df.columns[1]} Distribution', fontsize=16, fontweight='bold', pad=20)
        
        # Make percentage text bold and larger
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(13)
        
        # Make labels bold and larger
        for text in texts:
            text.set_fontweight('bold')
            text.set_fontsize(13)
            
    elif chart_type == "line":
        # Line chart
        plt.plot(df[df.columns[0]], df[df.columns[1]], marker='o', linewidth=2, markersize=8, color='steelblue')
        plt.xlabel(df.columns[0], fontsize=13, fontweight='bold')
        plt.ylabel(df.columns[1], fontsize=13, fontweight='bold')
        plt.title('Data Trend', fontsize=13, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
    elif chart_type == "histogram":
        # Histogram with larger fonts
        plt.hist(df[df.columns[0]], bins=10, color='steelblue', edgecolor='black', alpha=0.7)
        plt.xlabel(df.columns[0], fontsize=13, fontweight='bold')
        plt.ylabel('Frequency', fontsize=13, fontweight='bold')
        plt.title('Distribution Histogram', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        
    else:
        # Default to bar chart
        ax = df.plot(kind='bar', x=df.columns[0], y=df.columns[1], legend=False, color='steelblue')
        
        plt.xlabel(df.columns[0], fontsize=13, fontweight='bold')
        plt.ylabel(df.columns[1], fontsize=13, fontweight='bold')
        plt.title('Data Distribution', fontsize=13, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for i, v in enumerate(df[df.columns[1]]):
            ax.text(i, v + 5, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()

    # Save to BytesIO instead of file
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)

    # Encode to Base64
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return {"graph_base64": img_base64}


def route_to_answer(state: State):
    if state["intent"] == "visualize" or state["intent"] == "viz_data":
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