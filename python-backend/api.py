# python-backend/api.py

import asyncio
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import from main.py
from main import (
    triage_agent,
    schedule_agent,
    networking_agent,
    relevance_guardrail,
    jailbreak_guardrail,
)

from shared_types import AirlineAgentContext
from database import db_client

from agents import (
    Agent,
    RunContextWrapper,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Conference Agent API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# REQUEST/RESPONSE MODELS
# =========================

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    account_number: Optional[str] = None

class ChatResponse(BaseModel):
    conversation_id: str
    current_agent: str
    messages: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    context: Dict[str, Any]
    agents: List[Dict[str, Any]]
    guardrails: List[Dict[str, Any]]
    customer_info: Optional[Dict[str, Any]] = None

class CustomerInfoResponse(BaseModel):
    customer: Optional[Dict[str, Any]] = None
    bookings: List[Dict[str, Any]] = []
    current_booking: Optional[Dict[str, Any]] = None

# =========================
# GLOBAL STATE MANAGEMENT
# =========================

# Store conversation states
conversations: Dict[str, Dict[str, Any]] = {}

def get_all_agents():
    """Get all available agents."""
    return [triage_agent, schedule_agent, networking_agent]

def serialize_agent(agent: Agent) -> Dict[str, Any]:
    """Serialize an agent for API response."""
    try:
        # Safely extract handoff names
        handoff_names = []
        if hasattr(agent, 'handoffs') and agent.handoffs:
            for handoff in agent.handoffs:
                if hasattr(handoff, 'target') and hasattr(handoff.target, 'name'):
                    handoff_names.append(handoff.target.name)
                elif hasattr(handoff, 'name'):
                    handoff_names.append(handoff.name)
                else:
                    # Try to get string representation
                    handoff_names.append(str(handoff))
        
        # Safely extract tool names
        tool_names = []
        if hasattr(agent, 'tools') and agent.tools:
            for tool in agent.tools:
                if hasattr(tool, '__name__'):
                    tool_names.append(tool.__name__)
                else:
                    tool_names.append(str(tool))
        
        return {
            "name": agent.name,
            "description": agent.handoff_description or "",
            "handoffs": handoff_names,
            "tools": tool_names,
            "input_guardrails": getattr(agent, 'input_guardrails', []) or [],
        }
    except Exception as e:
        logger.error(f"Error serializing agent {agent.name}: {e}")
        return {
            "name": agent.name,
            "description": getattr(agent, 'handoff_description', '') or "",
            "handoffs": [],
            "tools": [],
            "input_guardrails": [],
        }

def serialize_event(event: Any) -> Dict[str, Any]:
    """Serialize an event for API response."""
    return {
        "id": str(uuid.uuid4()),
        "type": getattr(event, 'type', 'unknown'),
        "agent": getattr(event, 'agent', 'unknown'),
        "content": str(getattr(event, 'content', '')),
        "timestamp": datetime.now().isoformat(),
        "metadata": getattr(event, 'metadata', {}),
    }

def serialize_guardrail_check(name: str, passed: bool, reasoning: str = "") -> Dict[str, Any]:
    """Serialize a guardrail check for API response."""
    return {
        "id": str(uuid.uuid4()),
        "name": name,
        "input": "",
        "reasoning": reasoning,
        "passed": passed,
        "timestamp": datetime.now().isoformat(),
    }

async def load_user_context(identifier: str) -> Optional[Dict[str, Any]]:
    """Load user context from database."""
    try:
        # Try to get user by registration ID first
        user = await db_client.get_user_by_registration_id(identifier)
        
        # If not found, try by QR code (user ID)
        if not user:
            user = await db_client.get_user_by_qr_code(identifier)
        
        if user:
            return {
                "customer": user,
                "bookings": [],  # Conference users don't have flight bookings
                "current_booking": None
            }
        
        return None
    except Exception as e:
        logger.error(f"Error loading user context: {e}")
        return None

async def execute_agent_with_tools(agent: Agent, context: AirlineAgentContext, message: str) -> str:
    """Execute agent with proper tool calling."""
    try:
        run_context = RunContextWrapper(context)
        
        # Check if message matches tool usage patterns for schedule agent
        if agent.name == "Schedule Agent":
            from main import get_conference_schedule_tool
            
            message_lower = message.lower()
            
            # Parse different types of schedule queries
            if any(word in message_lower for word in ["events", "sessions", "schedule", "find", "show", "what"]):
                # Check for date patterns
                if "july 15" in message_lower or "2025-07-15" in message_lower or "15th" in message_lower:
                    result = await get_conference_schedule_tool(run_context, conference_date="2025-07-15")
                elif "july 16" in message_lower or "2025-07-16" in message_lower or "16th" in message_lower:
                    result = await get_conference_schedule_tool(run_context, conference_date="2025-07-16")
                elif "september" in message_lower or "sept" in message_lower:
                    # No events in September, but search anyway
                    result = await get_conference_schedule_tool(run_context, conference_date="2025-09-01")
                # Check for speaker names
                elif "alice" in message_lower:
                    result = await get_conference_schedule_tool(run_context, speaker_name="Alice Wonderland")
                elif "bob" in message_lower:
                    result = await get_conference_schedule_tool(run_context, speaker_name="Bob The Builder")
                elif "charlie" in message_lower:
                    result = await get_conference_schedule_tool(run_context, speaker_name="Charlie Chaplin")
                elif "diana" in message_lower:
                    result = await get_conference_schedule_tool(run_context, speaker_name="Diana Prince")
                # Check for topics
                elif "ai" in message_lower or "artificial intelligence" in message_lower:
                    result = await get_conference_schedule_tool(run_context, topic="AI")
                elif "cloud" in message_lower:
                    result = await get_conference_schedule_tool(run_context, topic="Cloud")
                elif "data" in message_lower:
                    result = await get_conference_schedule_tool(run_context, topic="Data")
                elif "web" in message_lower:
                    result = await get_conference_schedule_tool(run_context, topic="Web")
                elif "security" in message_lower or "cybersecurity" in message_lower:
                    result = await get_conference_schedule_tool(run_context, topic="Security")
                # Check for tracks
                elif "ai & ml" in message_lower or "ai ml" in message_lower or "machine learning" in message_lower:
                    result = await get_conference_schedule_tool(run_context, track_name="AI & ML")
                elif "data science" in message_lower:
                    result = await get_conference_schedule_tool(run_context, track_name="Data Science")
                elif "cloud computing" in message_lower:
                    result = await get_conference_schedule_tool(run_context, track_name="Cloud Computing")
                elif "web development" in message_lower:
                    result = await get_conference_schedule_tool(run_context, track_name="Web Development")
                elif "cybersecurity" in message_lower:
                    result = await get_conference_schedule_tool(run_context, track_name="Cybersecurity")
                elif "product management" in message_lower:
                    result = await get_conference_schedule_tool(run_context, track_name="Product Management")
                elif "startup" in message_lower or "entrepreneurship" in message_lower:
                    result = await get_conference_schedule_tool(run_context, track_name="Startup & Entrepreneurship")
                # Check for rooms
                elif "grand ballroom" in message_lower:
                    result = await get_conference_schedule_tool(run_context, conference_room_name="Grand Ballroom")
                elif "executive suite" in message_lower:
                    result = await get_conference_schedule_tool(run_context, conference_room_name="Executive Suite 1")
                elif "breakout room" in message_lower:
                    result = await get_conference_schedule_tool(run_context, conference_room_name="Breakout Room A")
                elif "innovation hub" in message_lower:
                    result = await get_conference_schedule_tool(run_context, conference_room_name="Innovation Hub")
                elif "networking lounge" in message_lower:
                    result = await get_conference_schedule_tool(run_context, conference_room_name="Networking Lounge")
                else:
                    # General search - get all sessions
                    result = await get_conference_schedule_tool(run_context)
                
                return result
        
        # Check if message matches tool usage patterns for networking agent
        elif agent.name == "Networking Agent":
            from main import (
                search_attendees_tool, 
                search_businesses_tool, 
                get_user_businesses_tool,
                display_business_form_tool,
                add_business_tool
            )
            
            message_lower = message.lower()
            
            # Check for business form submission (structured data)
            if "i want to add my business with the following details:" in message_lower:
                # Parse business details from the message
                lines = message.split('\n')
                business_data = {}
                
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower()
                        value = value.strip()
                        
                        if 'company name' in key:
                            business_data['company_name'] = value
                        elif 'industry sector' in key:
                            business_data['industry_sector'] = value
                        elif 'sub-sector' in key or 'sub sector' in key:
                            business_data['sub_sector'] = value
                        elif 'location' in key:
                            business_data['location'] = value
                        elif 'position title' in key:
                            business_data['position_title'] = value
                        elif 'legal structure' in key:
                            business_data['legal_structure'] = value
                        elif 'establishment year' in key:
                            business_data['establishment_year'] = value
                        elif 'products/services' in key or 'products or services' in key:
                            business_data['products_or_services'] = value
                        elif 'brief description' in key:
                            business_data['brief_description'] = value
                        elif 'website' in key:
                            business_data['website'] = value
                
                # Call add_business_tool with parsed data
                if len(business_data) >= 8:  # Ensure we have most required fields
                    result = await add_business_tool(
                        run_context,
                        company_name=business_data.get('company_name', ''),
                        industry_sector=business_data.get('industry_sector', ''),
                        sub_sector=business_data.get('sub_sector', ''),
                        location=business_data.get('location', ''),
                        position_title=business_data.get('position_title', ''),
                        legal_structure=business_data.get('legal_structure', ''),
                        establishment_year=business_data.get('establishment_year', ''),
                        products_or_services=business_data.get('products_or_services', ''),
                        brief_description=business_data.get('brief_description', ''),
                        website=business_data.get('website')
                    )
                    return result
            
            # Business-related queries
            if any(word in message_lower for word in ["business", "company", "companies"]):
                if "healthcare" in message_lower:
                    result = await search_businesses_tool(run_context, sector="Healthcare")
                elif "it" in message_lower or "technology" in message_lower:
                    result = await search_businesses_tool(run_context, sector="Technology")
                elif "finance" in message_lower:
                    result = await search_businesses_tool(run_context, sector="Finance")
                elif "manufacturing" in message_lower:
                    result = await search_businesses_tool(run_context, sector="Manufacturing")
                elif "mumbai" in message_lower:
                    result = await search_businesses_tool(run_context, location="Mumbai")
                elif "chennai" in message_lower:
                    result = await search_businesses_tool(run_context, location="Chennai")
                elif "delhi" in message_lower:
                    result = await search_businesses_tool(run_context, location="Delhi")
                elif "bangalore" in message_lower or "bengaluru" in message_lower:
                    result = await search_businesses_tool(run_context, location="Bangalore")
                elif "my business" in message_lower or "tell me about my business" in message_lower:
                    result = await get_user_businesses_tool(run_context)
                elif "add" in message_lower and "business" in message_lower:
                    result = await display_business_form_tool(run_context)
                else:
                    # General business search
                    result = await search_businesses_tool(run_context)
                
                return result
            
            # Attendee-related queries
            elif any(word in message_lower for word in ["attendee", "attendees", "people", "participant", "participants"]):
                if "chennai" in message_lower:
                    result = await search_attendees_tool(run_context, name="Chennai")
                elif "mumbai" in message_lower:
                    result = await search_attendees_tool(run_context, name="Mumbai")
                elif "delhi" in message_lower:
                    result = await search_attendees_tool(run_context, name="Delhi")
                elif "bangalore" in message_lower or "bengaluru" in message_lower:
                    result = await search_attendees_tool(run_context, name="Bangalore")
                elif "all" in message_lower:
                    result = await search_attendees_tool(run_context, limit=20)
                else:
                    result = await search_attendees_tool(run_context, limit=10)
                
                return result
        
        # If no specific tool pattern matched, return a helpful response
        if agent.name == "Schedule Agent":
            return """I can help you find conference schedule information. You can ask me about:

• Sessions by date - "What's happening on July 15th?" or "Events on July 16th"
• Sessions by speaker - "Show me sessions by Alice Wonderland"
• Sessions by topic - "Find AI sessions" or "Show me Cloud sessions"
• Sessions by room - "What's in the Grand Ballroom?"
• Sessions by track - "Show me Data Science track" or "AI & ML sessions"

What specific schedule information are you looking for?"""
        
        elif agent.name == "Networking Agent":
            return """I can help you with networking and business connections. You can ask me to:

• Find attendees - "Find attendees from Chennai" or "Show me all attendees"
• Search businesses - "Find healthcare businesses" or "Show me IT companies"
• Get business info - "Show me businesses in Mumbai" or "Tell me about my business"
• Add your business - "I want to add my business"

What networking assistance do you need?"""
        
        else:
            return """I'm your conference assistant for Business Conference 2025. I can help you with:

🗓️ Conference Schedule - Find sessions, speakers, timings, and rooms
🤝 Networking - Connect with attendees and explore business opportunities

What would you like to know about the conference?"""
            
    except Exception as e:
        logger.error(f"Error executing agent {agent.name}: {e}")
        return f"I'm having trouble processing your request. Please try again or rephrase your question."

# =========================
# API ENDPOINTS
# =========================

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint for conference assistance."""
    try:
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Load or create conversation state
        if conversation_id in conversations:
            conversation = conversations[conversation_id]
            context = AirlineAgentContext(**conversation["context"])
            current_agent_name = conversation["current_agent"]
            messages = conversation.get("messages", [])
            events = conversation.get("events", [])
        else:
            # Initialize new conversation
            context = AirlineAgentContext(
                is_conference_attendee=True,
                conference_name="Business Conference 2025"
            )
            current_agent_name = "Triage Agent"
            messages = []
            events = []
            
            # Load user context if account_number provided
            if request.account_number:
                user_info = await load_user_context(request.account_number)
                if user_info and user_info["customer"]:
                    customer = user_info["customer"]
                    context.passenger_name = customer.get("name")
                    context.customer_id = customer.get("id")
                    context.account_number = str(customer.get("account_number", ""))
                    context.customer_email = customer.get("email")
                    context.is_conference_attendee = customer.get("is_conference_attendee", True)
                    context.conference_name = customer.get("conference_name", "Business Conference 2025")
                    context.user_location = customer.get("location")
                    context.user_registration_id = str(customer.get("registration_id", ""))
                    context.user_conference_package = customer.get("conference_package")
                    context.user_primary_stream = customer.get("primary_stream")
                    context.user_secondary_stream = customer.get("secondary_stream")
                    context.user_company_name = customer.get("company")

        # Find current agent
        all_agents = get_all_agents()
        current_agent = next((a for a in all_agents if a.name == current_agent_name), triage_agent)
        
        # Process the message if provided
        if request.message.strip():
            # Check guardrails
            guardrail_results = []
            
            # Check relevance guardrail
            try:
                relevance_result = await relevance_guardrail(RunContextWrapper(context), request.message)
                guardrail_results.append(serialize_guardrail_check(
                    "relevance_guardrail", 
                    relevance_result.is_relevant, 
                    relevance_result.reasoning
                ))
                
                if not relevance_result.is_relevant:
                    response_message = "Sorry, I can only answer questions related to conference assistance."
                    messages.append({
                        "content": response_message,
                        "agent": current_agent.name
                    })
                    
                    # Save conversation state
                    conversations[conversation_id] = {
                        "context": context.model_dump(),
                        "current_agent": current_agent.name,
                        "messages": messages,
                        "events": events,
                    }
                    
                    return ChatResponse(
                        conversation_id=conversation_id,
                        current_agent=current_agent.name,
                        messages=[{"content": response_message, "agent": current_agent.name}],
                        events=[],
                        context=context.model_dump(),
                        agents=[serialize_agent(a) for a in all_agents],
                        guardrails=guardrail_results,
                        customer_info=await load_user_context(request.account_number) if request.account_number else None
                    )
            except Exception as e:
                logger.error(f"Error in relevance guardrail: {e}")
            
            # Check jailbreak guardrail
            try:
                jailbreak_result = await jailbreak_guardrail(RunContextWrapper(context), request.message)
                guardrail_results.append(serialize_guardrail_check(
                    "jailbreak_guardrail", 
                    jailbreak_result.is_safe, 
                    jailbreak_result.reasoning
                ))
                
                if not jailbreak_result.is_safe:
                    response_message = "Sorry, I can only answer questions related to conference assistance."
                    messages.append({
                        "content": response_message,
                        "agent": current_agent.name
                    })
                    
                    # Save conversation state
                    conversations[conversation_id] = {
                        "context": context.model_dump(),
                        "current_agent": current_agent.name,
                        "messages": messages,
                        "events": events,
                    }
                    
                    return ChatResponse(
                        conversation_id=conversation_id,
                        current_agent=current_agent.name,
                        messages=[{"content": response_message, "agent": current_agent.name}],
                        events=[],
                        context=context.model_dump(),
                        agents=[serialize_agent(a) for a in all_agents],
                        guardrails=guardrail_results,
                        customer_info=await load_user_context(request.account_number) if request.account_number else None
                    )
            except Exception as e:
                logger.error(f"Error in jailbreak guardrail: {e}")
            
            # Route to appropriate agent based on message content
            message_lower = request.message.lower()
            
            # Check for business form submission first
            if "i want to add my business with the following details:" in message_lower:
                current_agent = networking_agent
            elif any(word in message_lower for word in ["schedule", "session", "speaker", "event", "track", "room", "date", "time", "july", "september", "find sessions", "show sessions"]):
                current_agent = schedule_agent
            elif any(word in message_lower for word in ["business", "attendee", "networking", "company", "people", "participant"]):
                current_agent = networking_agent
            else:
                current_agent = triage_agent
            
            # Execute agent with tools
            try:
                response_message = await execute_agent_with_tools(current_agent, context, request.message)
                
                messages.append({
                    "content": response_message,
                    "agent": current_agent.name
                })
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                response_message = "I'm having trouble processing your request. Please try again."
                messages.append({
                    "content": response_message,
                    "agent": current_agent.name
                })
        
        # Save conversation state
        conversations[conversation_id] = {
            "context": context.model_dump(),
            "current_agent": current_agent.name,
            "messages": messages,
            "events": events,
        }
        
        # Get customer info
        customer_info = None
        if request.account_number:
            customer_info = await load_user_context(request.account_number)
        
        return ChatResponse(
            conversation_id=conversation_id,
            current_agent=current_agent.name,
            messages=messages[-1:] if messages else [],  # Return only the latest message
            events=[],  # Events would be populated by the full agent system
            context=context.model_dump(),
            agents=[serialize_agent(a) for a in all_agents],
            guardrails=[],  # Guardrails would be populated by the full agent system
            customer_info=customer_info
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/{identifier}", response_model=Dict[str, Any])
async def get_user_info(identifier: str):
    """Get user information by registration ID or QR code."""
    try:
        # Try to get user by registration ID first
        user = await db_client.get_user_by_registration_id(identifier)
        
        # If not found, try by QR code (user ID)
        if not user:
            user = await db_client.get_user_by_qr_code(identifier)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching user info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/customer/{account_number}", response_model=Dict[str, Any])
async def get_customer_info(account_number: str):
    """Get customer information by account number (for backward compatibility)."""
    try:
        customer = await db_client.get_customer_by_account_number(account_number)
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found")
        return customer
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching customer info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/booking/{confirmation_number}", response_model=Dict[str, Any])
async def get_booking_info(confirmation_number: str):
    """Get booking information by confirmation number."""
    try:
        booking = await db_client.get_booking_by_confirmation(confirmation_number)
        if not booking:
            raise HTTPException(status_code=404, detail="Booking not found")
        return booking
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching booking info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)