# Agent Switching with LiveKit

Previously I worked with LiveKit to build a voice AI agent, but had only one agent handling everything from greeting to booking. This isn't the most optimal approach. To learn how complex AI workflows are built in real industry agents, **agent switching** is used.

The idea: a general agent starts the conversation, and calls a specialized agent whenever required instead of one agent trying to do everything itself.

Today I learned about this and implemented it.

## Step 1: Basic version with simulated data

First I tried it with a simple, simulated setup — `simple_agent_switching.py`.

A **Receptionist agent** starts the conversation and answers general questions. The moment the caller wants to book, it calls a **Booking agent**, which takes over and asks for the details. No real backend here, just simulated availability and confirmation — the goal was to get the switching mechanism itself working first.

## Step 2: Extended onto gaming zone booking agent

Once the basic version worked, I extended the pattern onto the gaming zone booking agent I had already built earlier — `agent_switching_real_backend.py`.

I added a **Receptionist agent** in front of existing booking agent. The receptionist handles greetings and FAQs (hours, pricing, stations), and hands off to the booking agent when the caller wants to book which still runs on my real backend: [gaming_zone_backend](https://github.com/Vedika-Sd/gaming_zone_backend).


Console log screenshot below shows the exact moment the handoff happens when the user says *"I want to book a slot,"* the receptionist calls `transfer_to_booking`, and control passes to the booking agent, which immediately asks for the caller's name.

<img width="947" height="278" alt="image" src="https://github.com/user-attachments/assets/eb8c119f-e863-4fe4-9480-4b403b84545e" />


Full run log is also in this repo — `agent-switching-test.log`.

## How to run

```bash
python simple_agent_switching.py console
# or
python agent_switching_real_backend.py console
```

Needs a `.env.local` with `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`.

## Where this stands

This is just one step after the generic voice AI pipeline — STT → LLM → TTS with tool calling. 
Agent switching is the first step toward making it actually agentic, but there's a lot more to learn, implement, and try from here.
