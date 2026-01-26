from graph import build_graph
import os

class MockLLM:
    def invoke(self, messages):
        content = messages[0].content
        if "Classify" in content:
            if "members by state" in content:
                return type('obj', (object,), {'content': 'members_by_state'})
            if "visualize" in content or "chart" in content:
                return type('obj', (object,), {'content': 'visualize'})
            return type('obj', (object,), {'content': 'cooperatives_total'})
        return type('obj', (object,), {'content': 'Mock response from LLM'})

def test_visualizer_integration():
    llm = MockLLM()
    graph = build_graph(llm)
    
    # 1. Test standard query
    print("Testing standard query...")
    res1 = graph.invoke({"question": "How many cooperatives are there?"})
    print(f"Intent: {res1.get('intent')}")
    print(f"Answer: {res1['answer']}")
    print(f"Graph URL in state: {res1.get('graph_url')}")
    print("-" * 20)
    
    # 2. Test visualization query (members by state - file strategy)
    print("Testing members by state visualization...")
    res2 = graph.invoke({"question": "Visualize the members by state"})
    print(f"Intent: {res2.get('intent')}")
    print(f"Answer: {res2['answer']}")
    if "graph_url" in res2 and res2["graph_url"]:
        print(f"Graph URL in state: {res2['graph_url']}")
        local_path = res2['graph_url'].replace("/static/", "static/")
        if os.path.exists(local_path):
            print(f"SUCCESS: Graph file exists at {local_path}")
        else:
            print(f"FAILURE: Graph file NOT found at {local_path}")
    print("-" * 20)
    
    # 3. Test mock visualization query (unknown visualization)
    print("Testing mock visualization...")
    res3 = graph.invoke({"question": "Show me a chart of random trends"})
    print(f"Intent: {res3.get('intent')}")
    print(f"Answer: {res3['answer']}")
    if "graph_url" in res3 and res3["graph_url"]:
        print(f"Graph URL in state: {res3['graph_url']}")
        local_path = res3['graph_url'].replace("/static/", "static/")
        if os.path.exists(local_path):
            print(f"SUCCESS: Mock graph file exists at {local_path}")
    print("-" * 20)

if __name__ == "__main__":
    test_visualizer_integration()
