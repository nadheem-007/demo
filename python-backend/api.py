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
    run_demo_loop,
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
    return {
        "name": agent.name,
        "description": agent.handoff_description or "",
        "handoffs": [h.target.name for h in agent.handoffs],
        "tools": [tool.__name__ for tool in agent.tools],
        "input_guardrails": agent.input_guardrails or [],
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
                    context.account_number = customer.get("account_number")
                    context.customer_email = customer.get("email")
                    context.is_conference_attendee = customer.get("is_conference_attendee", True)
                    context.conference_name = customer.get("conference_name", "Business Conference 2025")
                    context.user_location = customer.get("location")
                    context.user_registration_id = customer.get("registration_id")
                    context.user_conference_package = customer.get("conference_package")
                    context.user_primary_stream = customer.get("primary_stream")
                    context.user_secondary_stream = customer.get("secondary_stream")
                    context.user_company_name = customer.get("company")

        # Find current agent
        all_agents = get_all_agents()
        current_agent = next((a for a in all_agents if a.name == current_agent_name), triage_agent)
        
        # Create run context
        run_context = RunContextWrapper(context)
        
        # Process the message if provided
        if request.message.strip():
            # Check guardrails
            guardrail_results = []
            
            # Check relevance guardrail
            try:
                relevance_result = await relevance_guardrail(run_context, request.message)
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
                jailbreak_result = await jailbreak_guardrail(run_context, request.message)
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
            
            # Process message with current agent
            try:
                # Get agent instructions
                instructions = current_agent.instructions(run_context, current_agent)
                
                # Simple message processing - in a real implementation, you'd use the full agent system
                response_message = f"I'm your conference assistant for {context.conference_name}. I can help you with:\n\n🗓️ Conference Schedule - Find sessions, speakers, timings, and rooms\n🤝 Networking - Connect with attendees and explore business opportunities\n\nWhat would you like to know about the conference?"
                
                # Check if this is a routing request
                message_lower = request.message.lower()
                
                if any(word in message_lower for word in ["schedule", "session", "speaker", "event", "track", "room", "date", "time"]):
                    current_agent = schedule_agent
                    response_message = "I can help you find conference schedule information. You can ask me about:\n\n• Sessions by speaker - \"Show me sessions by Alice Wonderland\"\n• Sessions by topic - \"Find AI sessions\"\n• Sessions by room - \"What's in the Grand Ballroom?\"\n• Sessions by track - \"Show me Data Science track\"\n• Sessions by date - \"What's happening on July 15th?\"\n\nWhat specific schedule information are you looking for?"
                elif any(word in message_lower for word in ["business", "attendee", "networking", "company", "people", "participant"]):
                    current_agent = networking_agent
                    response_message = "I can help you with networking and business connections. You can ask me to:\n\n• Find attendees - \"Find attendees from Chennai\" or \"Show me all attendees\"\n• Search businesses - \"Find healthcare businesses\" or \"Show me IT companies\"\n• Add your business - \"I want to add my business\"\n• Get business info - \"Show me businesses in Mumbai\"\n\nWhat networking assistance do you need?"
                
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