---
layout: post
title: Building a Local Agentic AI system
subtitle: With langgraph and ollama
gh-repo: nazpe/Thesis
gh-badge: [star, follow]
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/thumb.png
share-img: /assets/img/path.jpg
comments: true
mathjax: true
tags: [text]
author: Nuno Pedrosa
---

## Building a Agentic AI for Email Management

In an age where our inboxes often feel like digital black holes, overflowing with unread messages and urgent requests, the idea of an intelligent assistant to cut through the noise is incredibly appealing. That's exactly the problem I set out to solve with my latest project: an agentic AI tool designed to analyze and condense information from your email account.

**The Problem: Email Overload**

We've all been there. Hundreds, if not thousands, of unread emails. Important updates buried under newsletters, promotions, and spam. Sifting through it all is a time sink, and the sheer volume can lead to missed opportunities or delayed responses. My goal was to create a system that could proactively understand what's in my inbox and present me with the crucial information, freeing up valuable time and mental energy.

**Introducing the Email AI Agent**

My solution is an AI agent built using `langchain` and `langgraph`, powered by a local Large Language Model (LLM). This agent it's an intelligent system capable of:

1.  **Listing Unread Emails:** It can quickly scan your inbox and provide a concise list of unread messages, including their sender, subject, and date.
2.  **Summarizing Email Content:** The real power lies here. Given an email's unique ID, the agent can generate a short, plain-text summary of its content, allowing you to grasp the essence of a message without opening it.

Imagine asking your email client, "What's new in my inbox?" and getting a bulleted list of subjects and senders, followed by "Summarize the email about the project deadline." – that's the kind of interaction this tool enables.

**Under the Hood: How It Works**

Let's dive a bit into the technical architecture.

**1. Local-First Approach with Ollama**

One of the key decisions for this project was to leverage a local LLM. While cloud-based LLMs like ChatGPT or Gemini are powerful, running a model locally offers several advantages, including enhanced privacy and reduced API costs. I used `Ollama` to host `qwen3:1.7b`, a compact yet capable model. This setup means your sensitive email data never leaves your machine to be processed by a third-party LLM provider.

First, you'd typically initialize your chat model like this:

```python
from langchain.chat_models import init_chat_model

CHAT_MODEL = 'qwen3:1.7b'
llm = init_chat_model(CHAT_MODEL, model_provider='ollama')
```

**2. Connecting to Your Inbox: IMAP**

To interact with the email server, I used `imap-client` and `imap_tools`. These libraries allow the agent to securely log in to your email account (using credentials stored safely in a `.env` file) and fetch email metadata or full content.

Connecting to the mailbox is a straightforward process:

```python
import os
from dotenv import load_dotenv
from imap_tools import MailBox

load_dotenv() # Load environment variables

IMAP_HOST = os.getenv('IMAP_HOST')
IMAP_USER = os.getenv('IMAP_USER')
IMAP_PASSWORD = os.getenv('IMAP_PASSWORD')
IMAP_FOLDER = 'INBOX'

def connect():
    mail_box = MailBox(IMAP_HOST)
    mail_box.login(IMAP_USER, IMAP_PASSWORD, initial_folder=IMAP_FOLDER)
    return mail_box
```

**3. The Agentic Core: LangChain and LangGraph**

This is where the "intelligence" comes in.

*   **`langchain`:** This framework helps connect the LLM with external tools. I defined two custom tools: `list_unread_emails()` and `summarize_email(uid)`. The `@tool` decorator from `langchain_core.tools` makes it easy to expose these functions to the LLM.

    Here's how the `list_unread_emails` tool is defined:

    ```python
    from langchain_core.tools import tool
    from imap_tools import AND
    import json

    @tool
    def list_unread_emails():
        """Return a bullet list of every UNREAD message's UID, subject, date and sender"""
        with connect() as mb:
            unread = list(mb.fetch(criteria=AND(seen=False), headers_only=True, mark_seen=False))

        if not unread:
            return 'You have no unread messages.'

        response = json.dumps([
            {
                'uid': mail.uid,
                'date': mail.date.astimezone().strftime('%Y-%m-%d %H:%M'),
                'subject': mail.subject,
                'sender': mail.from_
            } for mail in unread
        ])
        return response
    ```

    And the `summarize_email` tool:

    ```python
    @tool
    def summarize_email(uid):
        """Summarize a single e-mail given it's IMAP UID. Return a short summary of the e-mails content / body in plain text."""
        with connect() as mb:
            mail = next(mb.fetch(AND(uid=uid), mark_seen=False), None)

            if not mail:
                return f'Could not summarize e-mail with UID {uid}.'

            prompt = (
                "Summarize this e-mail concisely:\n\n"
                f"Subject: {mail.subject}\n"
                f"Sender: {mail.from_}\n"
                f"Date: {mail.date}\n\n"
                f"{mail.text or mail.html}"
            )
            # 'raw_llm' is another instance of init_chat_model for summarization
            return raw_llm.invoke(prompt).content
    ```
    Once defined, these tools are bound to the LLM:
    ```python
    llm = llm.bind_tools([list_unread_emails, summarize_email])
    ```

*   **`langgraph`:** This library is crucial for building robust, multi-step agentic workflows. It allows the AI to decide *when* to use which tool and how to sequence its actions.

    *   **LLM Node:** The core reasoning component. The LLM processes your request and decides if it needs to use a tool or if it can answer directly.
    *   **Tool Node:** If the LLM decides to use a tool (e.g., to list emails or summarize one), this node executes the Python function associated with that tool.
    *   **Router:** After the LLM's response, the router checks if a tool was called. If so, it routes the flow back to the `tools_node` to execute it; otherwise, the conversation ends.

    The graph definition ties these components together:

    ```python
    from langgraph.prebuilt import ToolNode
    from langgraph.graph import StateGraph, START, END
    from typing import TypedDict

    class ChatState(TypedDict):
        messages: list

    # LLM node: invokes the LLM
    def llm_node(state):
        response = llm.invoke(state['messages'])
        return {'messages': state['messages'] + [response]}

    # Router: decides whether to use tools or end
    def router(state):
        last_message = state['messages'][-1]
        return 'tools' if getattr(last_message, 'tool_calls', None) else 'end'

    # Tool node: executes tools
    tool_node = ToolNode([list_unread_emails, summarize_email])
    def tools_node(state):
        result = tool_node.invoke(state)
        return {'messages': state['messages'] + result['messages']}

    # Build the graph
    builder = StateGraph(ChatState)
    builder.add_node('llm', llm_node)
    builder.add_node('tools', tools_node)
    builder.add_edge(START, 'llm')
    builder.add_edge('tools', 'llm')
    builder.add_conditional_edges('llm', router, {'tools': 'tools', 'end': END})

    graph = builder.compile()
    ```

    Here's a simplified visual of the agent's flow:

![Extraction Pipeline](https://github.com/user-attachments/assets/5c0bef95-1ce1-4d25-a102-fa8be522d88f){: .mx-auto.d-block :}

![Extraction Pipeline](https://github.com/user-attachments/assets/78481c50-3f61-437a-9c82-58eaf1cbc9f7){: .mx-auto.d-block :}

![Extraction Pipeline](https://github.com/user-attachments/assets/1dc6c0e4-977c-4e0f-906a-b618b7604e18){: .mx-auto.d-block :}

![Extraction Pipeline](https://github.com/user-attachments/assets/7e6f3461-d63e-4260-918d-164308a8a11b){: .mx-auto.d-block :}

**The Impact: A Smarter Inbox**

This project demonstrates the power of agentic AI in a practical, everyday scenario. Instead of being reactive to a constant flood of emails, you can proactively query your inbox, getting the information you need when you need it.

**Future Possibilities**

This is just the beginning! This agent could be extended to:

*   **Prioritize emails:** Identify and highlight messages from key contacts or about specific topics.
*   **Draft replies:** Generate preliminary responses based on the email content.
*   **Handle actions:** Archive, flag, or move emails based on your instructions.
*   **Integrate with other tools:** Connect with calendars or task managers to create events or to-dos directly from emails.

