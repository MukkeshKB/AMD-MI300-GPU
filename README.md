# Scheduling assistant using AMD MI300 GPU

Introduction:

This repository contains the final implementation for the Agentic AI Scheduling Assistant. The core of the solution is within the Submission.ipynb notebook, which runs a complete, autonomous scheduling service.

The assistant listens for meeting requests via an API endpoint, parses the request using a Deepseek Large Language Model (LLM) model comprising 7 billion parameters, checks the Google Calendars of all participants for availability, and finds a suitable time slot—all without human intervention.

Core Features:

This solution is built around the principles of Agentic AI to create a truly autonomous scheduling experience.

AI-Powered Email Parsing: Utilizes a sophisticated two-step "parse-and-judge" process with the deepseek-llm-7b-chat model. This ensures high accuracy in extracting key details like participants, meeting duration, and time constraints from unstructured, natural language text.

Automated Calendar Scheduling: Seamlessly integrates with the Google Calendar API to fetch real-time availability for all attendees. It intelligently merges busy slots to build a complete picture of shared free time.

Robust Time Handling: The scheduling engine can understand a wide range of relative time expressions, such as "tomorrow," "next Wednesday," or "in 2 weeks." It also includes logic to automatically avoid scheduling meetings on weekends, shifting them to the next working day.

API-Driven Service: The entire service is wrapped in a Flask web server, exposing a simple yet powerful /receive endpoint to accept new scheduling requests in JSON format.

Resilient and Debuggable: The code includes robust error handling, especially for the Google Calendar integration. If an attendee's authentication token is missing or expired, the system provides clear, actionable error messages to help diagnose the issue quickly.

Setup & Pre-Requisites:

To run this notebook successfully, please ensure the following prerequisites are met in your environment.

1. vLLM Server
   
A vLLM inference server must be running and accessible with the deepseek-llm-7b-chat model. Use the following command in a new terminal to launch the server:

```
HIP_VISIBLE_DEVICES=0 vllm serve /home/user/Models/deepseek-ai/deepseek-llm-7b-chat \
        --gpu-memory-utilization 0.9 \
        --swap-space 16 \
        --disable-log-requests \
        --dtype float16 \
        --max-model-len 2048 \
        --tensor-parallel-size 1 \
        --host 0.0.0.0 \
        --port 3000 \
        --num-scheduler-steps 10 \
        --max-num-seqs 128 \
        --max-num-batched-tokens 2048 \
        --max-model-len 2048 \
        --distributed-executor-backend "mp"
```

2. Google Calendar API Tokens
Valid Google Calendar authentication tokens are required for each potential attendee.

The token files must be named after the user's email prefix (e.g., userone.token).

These files must be placed in a directory named Keys that is located one level above the Submission.ipynb notebook (i.e., at ../Keys/).

3. Python Dependencies
Ensure all necessary Python libraries are installed. The primary libraries used are:

- flask
- google-api-python-client
- google-auth-oauthlib
- openai
- pytz

Input JSON format:

```json
{
    "Request_id": "6118b54f-907b-4451-8d48-dd13d76033a5",
    "Datetime": "19-07-2025T12:34:55",
    "Location": "IISc Bangalore",
    "From": "userone.amd@gmail.com",
    "Attendees": [
        {
            "email": "usertwo.amd@gmail.com"
        },
        {
            "email": "userthree.amd@gmail.com"
        }
    ],
    "Subject": "Agentic AI Project Status Update",
    "EmailContent": "Hi team, let's meet on Thursday for 30 minutes to discuss the status of Agentic AI Project."
}
```

Output JSON format:

```json
{
    "Request_id": "6118b54f-907b-4451-8d48-dd13d76033a5",
    "Datetime": "19-07-2025T12:34:55",
    "Location": "IISc Bangalore",
    "From": "userone.amd@gmail.com",
    "Attendees": [
        {
            "email": "userone.amd@gmail.com",
            "events": [
                {
                    "StartTime": "2025-07-24T10:00:00+05:30",
                    "EndTime": "2025-07-24T10:30:00+05:30",
                    "NumAttendees": 3,
                    "Attendees": [
                        "userone.amd@gmail.com",
                        "usertwo.amd@gmail.com",
                        "userthree.amd@gmail.com"
                    ],
                    "Summary": "Team Meet"
                },
                {
                    "StartTime": "2025-07-24T10:30:00+05:30",
                    "EndTime": "2025-07-24T11:00:00+05:30",
                    "NumAttendees": 3,
                    "Attendees": [
                        "userone.amd@gmail.com",
                        "usertwo.amd@gmail.com",
                        "userthree.amd@gmail.com"
                    ],
                    "Summary": "Agentic AI Project Status Update"
                }
            ]
        },
        {
            "email": "usertwo.amd@gmail.com",
            "events": [
                {
                    "StartTime": "2025-07-24T10:00:00+05:30",
                    "EndTime": "2025-07-24T10:30:00+05:30",
                    "NumAttendees": 3,
                    "Attendees": [
                        "userone.amd@gmail.com",
                        "usertwo.amd@gmail.com",
                        "userthree.amd@gmail.com"
                    ],
                    "Summary": "Team Meet"
                },
                {
                    "StartTime": "2025-07-24T10:30:00+05:30",
                    "EndTime": "2025-07-24T11:00:00+05:30",
                    "NumAttendees": 3,
                    "Attendees": [
                        "userone.amd@gmail.com",
                        "usertwo.amd@gmail.com",
                        "userthree.amd@gmail.com"
                    ],
                    "Summary": "Agentic AI Project Status Update"
                }
            ]
        },
        {
            "email": "userthree.amd@gmail.com",
            "events": [
                {
                    "StartTime": "2025-07-24T10:00:00+05:30",
                    "EndTime": "2025-07-24T10:30:00+05:30",
                    "NumAttendees": 3,
                    "Attendees": [
                        "userone.amd@gmail.com",
                        "usertwo.amd@gmail.com",
                        "userthree.amd@gmail.com"
                    ],
                    "Summary": "Team Meet"
                },
                {
                    "StartTime": "2025-07-24T13:00:00+05:30",
                    "EndTime": "2025-07-24T14:00:00+05:30",
                    "NumAttendees": 1,
                    "Attendees": [
                        "SELF"
                    ],
                    "Summary": "Lunch with Customers"
                },
                {
                    "StartTime": "2025-07-24T10:30:00+05:30",
                    "EndTime": "2025-07-24T11:00:00+05:30",
                    "NumAttendees": 3,
                    "Attendees": [
                        "userone.amd@gmail.com",
                        "usertwo.amd@gmail.com",
                        "userthree.amd@gmail.com"
                    ],
                    "Summary": "Agentic AI Project Status Update"
                }
            ]
        }
    ],
    "Subject": "Agentic AI Project Status Update",
    "EmailContent": "Hi team, let's meet on Thursday for 30 minutes to discuss the status of Agentic AI Project.",
    "EventStart": "2025-07-24T10:30:00+05:30",
    "EventEnd": "2025-07-24T11:00:00+05:30",
    "Duration_mins": "30",
    "MetaData": {}
}
```









