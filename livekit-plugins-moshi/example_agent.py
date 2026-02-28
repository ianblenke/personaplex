"""
Example LiveKit agent using Moshi/PersonaPlex for real-time voice AI.

Prerequisites:
    1. A running Moshi/PersonaPlex server (e.g. via docker-compose)
    2. A running LiveKit server (https://docs.livekit.io/home/self-hosting/local/)
    3. Install this plugin:
        cd livekit-plugins-moshi && pip install -e .

Usage:
    export LIVEKIT_URL=ws://localhost:7880
    export LIVEKIT_API_KEY=devkey
    export LIVEKIT_API_SECRET=secret

    python example_agent.py dev

    Then connect a client (e.g. https://agents-playground.livekit.io/)
"""

from livekit.agents import Agent, AgentSession, WorkerOptions, cli
from livekit.plugins.moshi import MoshiRealtimeModel


class MoshiAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful conversational assistant.",
        )


async def entrypoint(ctx: cli.JobContext) -> None:
    await ctx.connect()

    session = AgentSession(
        llm=MoshiRealtimeModel(
            server_url="ws://localhost:8998",
            voice="NATM1",
            text_prompt="You are a friendly and helpful assistant. "
            "Keep responses concise and natural.",
            text_temperature=0.7,
            audio_temperature=0.8,
        ),
    )

    await session.start(agent=MoshiAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
