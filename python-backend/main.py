# python-backend/main.py

from __future__ import annotations as _annotations

import asyncio
import logging
import os
from typing import Optional, List, Dict, Any
from datetime import date, datetime

from agents import (
    Agent,
    RunContextWrapper,
    function_tool,
    handoff,
    run_agent_loop,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from shared_types import AirlineAgentContext, RelevanceOutput, JailbreakOutput
from database import db_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# TOOLS (Conference-specific)
# =========================

@function_tool(
    name_override="get_conference_schedule",
    description_override="Get conference schedule information by speaker, topic, room, track, or date."
)
async def get_conference_schedule_tool(
    context: RunContextWrapper[AirlineAgentContext],
    speaker_name: Optional[str] = None,
    topic: Optional[str] = None,
    conference_room_name: Optional[str] = None,
    track_name: Optional[str] = None,
    conference_date: Optional[str] = None
) -> str:
    """Get conference schedule information based on various filters."""
    try:
        # Convert date string to date object if provided
        parsed_date = None
        if conference_date:
            try:
                parsed_date = datetime.strptime(conference_date, "%Y-%m-%d").date()
            except ValueError:
                return f"Invalid date format: {conference_date}. Please use YYYY-MM-DD format."

        # Get schedule from database
        schedule = await db_client.get_conference_schedule(
            speaker_name=speaker_name,
            topic=topic,
            conference_room_name=conference_room_name,
            track_name=track_name,
            conference_date=parsed_date
        )

        if not schedule:
            filters = []
            if speaker_name: filters.append(f"speaker '{speaker_name}'")
            if topic: filters.append(f"topic '{topic}'")
            if conference_room_name: filters.append(f"room '{conference_room_name}'")
            if track_name: filters.append(f"track '{track_name}'")
            if conference_date: filters.append(f"date '{conference_date}'")
            
            filter_text = " and ".join(filters) if filters else "your criteria"
            return f"No conference sessions found for {filter_text}."

        # Format the schedule information
        result = f"Found {len(schedule)} conference session(s):\n\n"
        
        for session in schedule:
            start_time = session.get('start_time', 'TBD')
            end_time = session.get('end_time', 'TBD')
            
            # Format datetime strings if they exist
            if isinstance(start_time, str) and 'T' in start_time:
                start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00')).strftime('%I:%M %p')
            if isinstance(end_time, str) and 'T' in end_time:
                end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00')).strftime('%I:%M %p')
            
            result += f"**{session.get('topic', 'Unknown Topic')}**\n"
            result += f"Speaker: {session.get('speaker_name', 'TBD')}\n"
            result += f"Time: {start_time} - {end_time}\n"
            result += f"Room: {session.get('conference_room_name', 'TBD')}\n"
            result += f"Track: {session.get('track_name', 'TBD')}\n"
            result += f"Date: {session.get('conference_date', 'TBD')}\n"
            
            if session.get('description'):
                result += f"Description: {session.get('description')}\n"
            
            result += "\n"

        return result

    except Exception as e:
        return f"Error retrieving conference schedule: {str(e)}"

@function_tool(
    name_override="search_attendees",
    description_override="Search for conference attendees by name or get all attendees."
)
async def search_attendees_tool(
    context: RunContextWrapper[AirlineAgentContext],
    name: Optional[str] = None,
    limit: int = 10
) -> str:
    """Search for conference attendees by name or get all attendees."""
    try:
        if name:
            # Search by name
            attendees = await db_client.get_user_details_by_name(name)
        else:
            # Get all attendees
            attendees = await db_client.get_all_attendees(limit=limit)

        if not attendees:
            search_text = f"named '{name}'" if name else "in the conference"
            return f"No attendees found {search_text}."

        # Format attendee information
        result = f"Found {len(attendees)} attendee(s):\n\n"
        
        for attendee in attendees:
            details = attendee.get('details', {})
            user_name = details.get('user_name') or f"{details.get('firstName', '')} {details.get('lastName', '')}".strip()
            
            result += f"**{user_name}**\n"
            
            if details.get('company'):
                result += f"Company: {details.get('company')}\n"
            if details.get('location'):
                result += f"Location: {details.get('location')}\n"
            if details.get('primary_stream'):
                result += f"Primary Stream: {details.get('primary_stream')}\n"
            if details.get('secondary_stream'):
                result += f"Secondary Stream: {details.get('secondary_stream')}\n"
            if details.get('conference_package'):
                result += f"Conference Package: {details.get('conference_package')}\n"
            
            result += "\n"

        return result

    except Exception as e:
        return f"Error searching attendees: {str(e)}"

@function_tool(
    name_override="search_businesses",
    description_override="Search for businesses by company name, sector, or location."
)
async def search_businesses_tool(
    context: RunContextWrapper[AirlineAgentContext],
    query: Optional[str] = None,
    sector: Optional[str] = None,
    location: Optional[str] = None
) -> str:
    """Search for businesses by various criteria."""
    try:
        businesses = await db_client.search_businesses(
            query=query,
            sector=sector,
            location=location
        )

        if not businesses:
            filters = []
            if query: filters.append(f"query '{query}'")
            if sector: filters.append(f"sector '{sector}'")
            if location: filters.append(f"location '{location}'")
            
            filter_text = " and ".join(filters) if filters else "your criteria"
            return f"No businesses found for {filter_text}."

        # Format business information
        result = f"Found {len(businesses)} business(es):\n\n"
        
        for business in businesses:
            details = business.get('details', {})
            
            result += f"**{details.get('companyName', 'Unknown Company')}**\n"
            
            if details.get('industrySector'):
                result += f"Industry: {details.get('industrySector')}\n"
            if details.get('subSector'):
                result += f"Sub-sector: {details.get('subSector')}\n"
            if details.get('location'):
                result += f"Location: {details.get('location')}\n"
            if details.get('briefDescription'):
                result += f"Description: {details.get('briefDescription')}\n"
            if details.get('productsOrServices'):
                result += f"Products/Services: {details.get('productsOrServices')}\n"
            if details.get('web'):
                result += f"Website: {details.get('web')}\n"
            
            result += "\n"

        return result

    except Exception as e:
        return f"Error searching businesses: {str(e)}"

@function_tool(
    name_override="get_user_businesses",
    description_override="Get all businesses for a specific user."
)
async def get_user_businesses_tool(
    context: RunContextWrapper[AirlineAgentContext],
    user_name: Optional[str] = None
) -> str:
    """Get all businesses for a specific user."""
    try:
        # If no user_name provided, use current user
        if not user_name:
            user_id = context.context.customer_id
            if not user_id:
                return "No user specified and no current user context available."
        else:
            # Search for user by name first
            users = await db_client.get_user_details_by_name(user_name)
            if not users:
                return f"No user found with name '{user_name}'."
            user_id = users[0].get('id')

        businesses = await db_client.get_user_businesses(user_id)

        if not businesses:
            user_text = user_name or "the current user"
            return f"No businesses found for {user_text}."

        # Format business information
        result = f"Found {len(businesses)} business(es) for {user_name or 'the current user'}:\n\n"
        
        for business in businesses:
            details = business.get('details', {})
            
            result += f"**{details.get('companyName', 'Unknown Company')}**\n"
            
            if details.get('industrySector'):
                result += f"Industry: {details.get('industrySector')}\n"
            if details.get('subSector'):
                result += f"Sub-sector: {details.get('subSector')}\n"
            if details.get('location'):
                result += f"Location: {details.get('location')}\n"
            if details.get('positionTitle'):
                result += f"Position: {details.get('positionTitle')}\n"
            if details.get('briefDescription'):
                result += f"Description: {details.get('briefDescription')}\n"
            if details.get('web'):
                result += f"Website: {details.get('web')}\n"
            
            result += "\n"

        return result

    except Exception as e:
        return f"Error retrieving user businesses: {str(e)}"

@function_tool(
    name_override="display_business_form",
    description_override="Display a business registration form for the user to fill out."
)
async def display_business_form_tool(
    context: RunContextWrapper[AirlineAgentContext]
) -> str:
    """Trigger the UI to show a business registration form."""
    return "DISPLAY_BUSINESS_FORM"

@function_tool(
    name_override="add_business",
    description_override="Add a new business for the current user."
)
async def add_business_tool(
    context: RunContextWrapper[AirlineAgentContext],
    company_name: str,
    industry_sector: str,
    sub_sector: str,
    location: str,
    position_title: str,
    legal_structure: str,
    establishment_year: str,
    products_or_services: str,
    brief_description: str,
    website: Optional[str] = None
) -> str:
    """Add a new business for the current user."""
    try:
        user_id = context.context.customer_id
        if not user_id:
            return "Unable to add business: No user context available."

        # Prepare business details
        business_details = {
            "companyName": company_name,
            "industrySector": industry_sector,
            "subSector": sub_sector,
            "location": location,
            "positionTitle": position_title,
            "legalStructure": legal_structure,
            "establishmentYear": establishment_year,
            "productsOrServices": products_or_services,
            "briefDescription": brief_description
        }
        
        if website:
            business_details["web"] = website

        # Add business to database
        success = await db_client.add_business(user_id, business_details)

        if success:
            return f"Successfully added business '{company_name}' to your profile. The business is now listed in the business directory and available for networking."
        else:
            return f"Failed to add business '{company_name}'. Please try again or contact support."

    except Exception as e:
        return f"Error adding business: {str(e)}"

@function_tool(
    name_override="get_organization_info",
    description_override="Get information about an organization."
)
async def get_organization_info_tool(
    context: RunContextWrapper[AirlineAgentContext],
    organization_id: Optional[str] = None
) -> str:
    """Get organization information."""
    try:
        # If no organization_id provided, use current user's organization
        if not organization_id:
            organization_id = context.context.get('organization_id')
            if not organization_id:
                return "No organization specified and no current organization context available."

        organization = await db_client.get_organization_details(organization_id)

        if not organization:
            return f"No organization found with ID '{organization_id}'."

        # Format organization information
        result = f"**Organization Information**\n\n"
        result += f"Name: {organization.get('name', 'Unknown')}\n"
        
        details = organization.get('details', {})
        if details:
            for key, value in details.items():
                if value:
                    result += f"{key.replace('_', ' ').title()}: {value}\n"

        return result

    except Exception as e:
        return f"Error retrieving organization information: {str(e)}"

# =========================
# GUARDRAILS
# =========================

@function_tool
async def relevance_guardrail(
    context: RunContextWrapper[AirlineAgentContext], user_input: str
) -> RelevanceOutput:
    """Check if user input is relevant to conference assistance."""
    # Simple keyword-based relevance check for conference topics
    conference_keywords = [
        "conference", "session", "speaker", "schedule", "attendee", "business", 
        "networking", "track", "room", "event", "presentation", "workshop",
        "company", "organization", "meeting", "agenda", "registration"
    ]
    
    user_input_lower = user_input.lower()
    is_relevant = any(keyword in user_input_lower for keyword in conference_keywords)
    
    # Also check for general greetings and questions
    general_keywords = ["hello", "hi", "help", "what", "how", "when", "where", "who", "can you"]
    is_general = any(keyword in user_input_lower for keyword in general_keywords)
    
    if is_relevant or is_general:
        return RelevanceOutput(
            reasoning="User input is related to conference assistance or is a general inquiry.",
            is_relevant=True
        )
    else:
        return RelevanceOutput(
            reasoning="User input does not appear to be related to conference assistance.",
            is_relevant=False
        )

@function_tool
async def jailbreak_guardrail(
    context: RunContextWrapper[AirlineAgentContext], user_input: str
) -> JailbreakOutput:
    """Check if user input contains jailbreak attempts."""
    jailbreak_patterns = [
        "ignore", "system", "prompt", "instruction", "override", "bypass",
        "pretend", "roleplay", "act as", "forget", "disregard"
    ]
    
    user_input_lower = user_input.lower()
    is_jailbreak = any(pattern in user_input_lower for pattern in jailbreak_patterns)
    
    return JailbreakOutput(
        reasoning="Detected potential jailbreak attempt." if is_jailbreak else "No jailbreak patterns detected.",
        is_safe=not is_jailbreak
    )

# =========================
# AGENTS
# =========================

def schedule_agent_instructions(
    run_context: RunContextWrapper[AirlineAgentContext], agent: Agent[AirlineAgentContext]
) -> str:
    ctx = run_context.context
    user_name = ctx.passenger_name or "[unknown]"
    return (
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "You are a Conference Schedule Agent. Help attendees find information about conference sessions, speakers, and schedules.\n"
        f"Current user: {user_name}\n"
        "You can help with:\n"
        "1. Finding sessions by speaker name, topic, room, track, or date\n"
        "2. Getting schedule information for specific time periods\n"
        "3. Providing details about conference rooms and tracks\n"
        "4. Answering questions about session timings and descriptions\n\n"
        "IMPORTANT: Always use the get_conference_schedule tool to search for sessions when users ask about:\n"
        "- Sessions, speakers, topics, rooms, tracks, or dates\n"
        "- Schedule information\n"
        "- Conference events\n"
        "- What's happening on specific dates\n"
        "- Who is speaking\n"
        "- Session details\n\n"
        "Examples of when to use the tool:\n"
        "- 'events on september 1st' -> use get_conference_schedule with conference_date='2025-09-01'\n"
        "- 'sessions by track' -> ask which track, then use get_conference_schedule with track_name\n"
        "- 'find sessions' -> use get_conference_schedule to show all sessions\n"
        "- 'AI sessions' -> use get_conference_schedule with topic='AI'\n"
        "- 'Alice Wonderland sessions' -> use get_conference_schedule with speaker_name='Alice Wonderland'\n\n"
        "Do not just provide generic responses. Always search the database for actual schedule information.\n"
        "If the user asks unrelated questions, transfer back to the triage agent."
    )

schedule_agent = Agent[AirlineAgentContext](
    name="Schedule Agent",
    model="groq/llama3-8b-8192",
    handoff_description="An agent to provide conference schedule information and help find sessions.",
    instructions=schedule_agent_instructions,
    tools=[get_conference_schedule_tool],
    handoffs=[],
)

def networking_agent_instructions(
    run_context: RunContextWrapper[AirlineAgentContext], agent: Agent[AirlineAgentContext]
) -> str:
    ctx = run_context.context
    user_name = ctx.passenger_name or "[unknown]"
    return (
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "You are a Networking Agent. Help attendees connect with other participants and explore business opportunities.\n"
        f"Current user: {user_name}\n"
        "You can help with:\n"
        "1. Finding other conference attendees by name or interests\n"
        "2. Searching for businesses by company name, industry sector, or location\n"
        "3. Getting information about specific users' businesses\n"
        "4. Helping users register their own businesses\n"
        "5. Providing organization information\n\n"
        "IMPORTANT: Always use the appropriate tools to search for actual data when users ask about:\n"
        "- Attendees, people, or participants\n"
        "- Businesses, companies, or organizations\n"
        "- Healthcare, IT, or any industry businesses\n"
        "- Businesses in specific locations\n"
        "- User's own businesses\n\n"
        "Available tools and when to use them:\n"
        "- search_attendees: When users ask about attendees, people, or participants\n"
        "- search_businesses: When users ask about businesses, companies, or specific industries/locations\n"
        "- get_user_businesses: When users ask about their own businesses or a specific person's businesses\n"
        "- display_business_form: ONLY when users explicitly want to add/register a new business\n"
        "- add_business: Used automatically when user submits business form\n"
        "- get_organization_info: When users ask about organization details\n\n"
        "Examples of when to use tools:\n"
        "- 'Find healthcare businesses' -> use search_businesses with sector='Healthcare'\n"
        "- 'Show me businesses in Mumbai' -> use search_businesses with location='Mumbai'\n"
        "- 'tell me about my business' -> use get_user_businesses\n"
        "- 'Find attendees from Chennai' -> use search_attendees with name containing Chennai info\n"
        "- 'I want to add my business' -> use display_business_form\n\n"
        "Do not just provide generic responses. Always search the database for actual information.\n"
        "Do not show the business form unless the user explicitly asks to add/register a business.\n"
        "If the user asks unrelated questions, transfer back to the triage agent."
    )

networking_agent = Agent[AirlineAgentContext](
    name="Networking Agent",
    model="groq/llama3-8b-8192",
    handoff_description="An agent to help with networking, finding attendees, and business connections.",
    instructions=networking_agent_instructions,
    tools=[
        search_attendees_tool,
        search_businesses_tool,
        get_user_businesses_tool,
        display_business_form_tool,
        add_business_tool,
        get_organization_info_tool
    ],
    handoffs=[],
)

def triage_instructions(
    run_context: RunContextWrapper[AirlineAgentContext], agent: Agent[AirlineAgentContext]
) -> str:
    ctx = run_context.context
    user_name = ctx.passenger_name or "[unknown]"
    conference_name = ctx.conference_name or "Business Conference 2025"
    
    return (
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        f"You are a Conference Triage Agent for {conference_name}. Your role is to understand what the user needs and route them to the appropriate specialist agent.\n"
        f"Current user: {user_name}\n\n"
        "Available specialist agents:\n"
        "1. **Schedule Agent** - For conference schedule, sessions, speakers, timings, rooms, tracks, and dates\n"
        "2. **Networking Agent** - For finding attendees, businesses, networking opportunities, and business registration\n\n"
        "Route users based on their requests:\n"
        "- Schedule/Session questions → Schedule Agent\n"
        "- Networking/Business questions → Networking Agent\n\n"
        "Examples:\n"
        "- 'events on september 1st' → Schedule Agent\n"
        "- 'find sessions' → Schedule Agent\n"
        "- 'AI sessions' → Schedule Agent\n"
        "- 'speaker information' → Schedule Agent\n"
        "- 'find healthcare businesses' → Networking Agent\n"
        "- 'show attendees' → Networking Agent\n"
        "- 'my business' → Networking Agent\n"
        "- 'add business' → Networking Agent\n\n"
        "For general greetings or unclear requests, provide a helpful overview of what you can assist with and ask for clarification.\n"
        "Always be welcoming and explain the conference assistance available."
    )

triage_agent = Agent[AirlineAgentContext](
    name="Triage Agent",
    model="groq/llama3-8b-8192",
    handoff_description="Main entry point for conference assistance - routes users to appropriate specialists.",
    instructions=triage_instructions,
    tools=[],
    handoffs=[handoff(schedule_agent), handoff(networking_agent)],
    input_guardrails=["relevance_guardrail", "jailbreak_guardrail"],
)

# Update handoffs for other agents to include triage
schedule_agent.handoffs = [handoff(triage_agent), handoff(networking_agent)]
networking_agent.handoffs = [handoff(triage_agent), handoff(schedule_agent)]

# =========================
# MAIN EXECUTION
# =========================

async def main():
    """Main function to run the conference agent system."""
    
    # Initialize context with conference-specific information
    context = AirlineAgentContext(
        is_conference_attendee=True,
        conference_name="Business Conference 2025"
    )
    
    # Define guardrails
    guardrails = {
        "relevance_guardrail": relevance_guardrail,
        "jailbreak_guardrail": jailbreak_guardrail,
    }
    
    # Run the agent loop
    await run_agent_loop(
        starting_agent=triage_agent,
        context=context,
        guardrails=guardrails,
    )

if __name__ == "__main__":
    asyncio.run(main())