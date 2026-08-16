"""
SIMPLE AGENT SWITCHING DEMO — LiveKit Agents
==============================================
Two agents:
  1. ReceptionistAgent — greets caller, answers basic FAQs, listens for booking intent
  2. BookingAgent       — takes over once caller wants to book, collects details, "confirms" a booking

Run locally with:
    python simple_agent_switching.py console
"""

import random
import string

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    cli,
    function_tool,
    inference,
)

load_dotenv(".env.local")


# AGENT 1: Receptionist

class ReceptionistAgent(Agent):
    """
    Handles general conversation and FAQs. The moment the caller shows
    intent to book, it calls `transfer_to_booking`, which hands control
    over to BookingAgent. This is the core "agent switching" mechanism —
    a function tool that RETURNS a different Agent instance.
    """

    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a friendly receptionist for a gaming zone.

You can answer general questions:
- Opening hours: 9 AM to 11 PM, every day
- Pricing: 200 rupees per hour per station
- Stations available: PC, PS5, Xbox, VR, Racing Simulator

Keep answers short — one to three sentences, plain spoken language, no lists
or markdown, since this is a voice conversation.

If the caller wants to book a slot, or clearly wants to make a reservation,
call the transfer_to_booking tool right away. Do not try to collect booking
details yourself — that is the booking agent's job."""
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions=(
                "Greet the caller warmly. Mention you can answer questions about "
                "the gaming zone or help them book a session. End by asking what "
                "they'd like to do."
            )
        )

    @function_tool(name="transfer_to_booking")
    async def transfer_to_booking(self, context: RunContext):
        """Call this the moment the caller wants to book a gaming slot or
        clearly expresses intent to make a reservation."""
        return BookingAgent()


# AGENT 2: Booking specialist

class BookingAgent(Agent):
    """
    Takes over once the caller wants to book. Collects details through
    normal conversation, then calls two simple tools: one to "check"
    availability (simulated), and one to "confirm" the booking (simulated).
    """

    def __init__(self) -> None:
        super().__init__(
            instructions="""You are the booking specialist for a gaming zone.
The caller has already been greeted by the receptionist — do not greet them
again, continue the conversation naturally.

Collect these details one question at a time:
- Full name
- Preferred date and time
- Which station: PC, PS5, Xbox, VR, or Racing Simulator
- Number of players

Once you have date, time, and station, call check_availability before
confirming anything. Only call create_booking after the caller has
explicitly confirmed the details out loud.

Keep replies short and conversational — one to three sentences, plain
spoken language, no lists or markdown."""
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions=(
                "Continue the conversation naturally — ask for the first "
                "piece of booking information (their name)."
            )
        )

    @function_tool(name="check_availability")
    async def check_availability(
        self, context: RunContext, date: str, time: str, station_type: str
    ) -> str:
        """Check if a station is available at the given date and time.

        Args:
            date: booking date, e.g. "2026-08-20"
            time: booking time, e.g. "18:00"
            station_type: PC, PS5, Xbox, VR, or Racing Simulator
        """
        # Simulated check — in a real system this would call your backend.
        is_available = random.choice([True, True, True, False])  # mostly available
        if is_available:
            return f"{station_type} is available on {date} at {time}."
        return f"{station_type} is fully booked on {date} at {time}. Suggest a different time or station."

    @function_tool(name="create_booking")
    async def create_booking(
        self,
        context: RunContext,
        player_name: str,
        date: str,
        time: str,
        station_type: str,
        number_of_players: int,
    ) -> str:
        """Confirm and finalize the booking after the caller has agreed to the details.

        Args:
            player_name: caller's full name
            date: booking date
            time: booking time
            station_type: PC, PS5, Xbox, VR, or Racing Simulator
            number_of_players: how many players
        """
        # Simulated booking — generates a fake confirmation code instead of
        # hitting a real database/API.
        code = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
        return (
            f"Booking confirmed for {player_name}, {number_of_players} player(s), "
            f"{station_type} on {date} at {time}. Confirmation code: {code}."
        )


# Entry point — the call ALWAYS starts with ReceptionistAgent
server = AgentServer()


@server.rtc_session(agent_name="gaming-zone-simple")
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="en-IN"),
        llm=inference.LLM(model="google/gemma-4-31b-it"),
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="638efaaa-4d0c-442e-b701-3fae16aad012",
            language="en-IN",
        ),
        vad=inference.VAD(),
    )

    await session.start(
        agent=ReceptionistAgent(),  # call always starts here
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(server)
