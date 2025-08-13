# 🎯 Personality Coach MCP Server - Puch AI Hackathon Project

**A comprehensive MBTI personality assessment and coaching platform built with Model Context Protocol (MCP)**

This project is a sophisticated personality coaching system that combines MBTI personality assessment, intelligent matching algorithms, and personalized coaching tools. Built for the Puch AI Hackathon, it demonstrates the power of MCP to create AI-driven personality insights and relationship coaching.

## 🌟 Project Overview

The Personality Coach MCP Server is a full-featured personality assessment and coaching platform that provides:

- **MBTI Personality Assessment**: Advanced 16-question quiz with statistical confidence scoring
- **Intelligent Matching System**: Algorithm-driven personality compatibility matching
- **Personalized Coaching**: Context-aware tips for communication and relationship building
- **Real-time Communication Tools**: Message translation and tone adaptation
- **Persistent User Profiles**: Complete user state management with PostgreSQL/Supabase

## 🛠 Technology Stack

### Core Technologies
- **Python 3.11+**: Modern async/await architecture
- **FastMCP**: High-performance MCP server framework
- **HTTPX**: Async HTTP client with connection pooling
- **Pydantic**: Type-safe data validation and serialization

### Authentication & Security
- **Bearer Token Authentication**: RSA-based JWT token validation
- **OAuth2 Support**: Industry-standard authentication protocols
- **Secure Key Management**: Environment-based configuration

### Database & Persistence
- **PostgreSQL**: Primary database via Supabase
- **PostgREST**: RESTful API layer for database operations
- **JSONB Storage**: Flexible schema for personality data
- **Automatic Timestamps**: Audit trail with triggers

### Containerization & Deployment
- **Docker**: Containerized deployment with multi-stage builds
- **Docker Compose**: Orchestrated development environment
- **Health Checks**: Built-in monitoring and logging
- **Environment Variables**: Configurable deployment settings

### Data Processing & Analysis
- **Beautiful Soup**: HTML parsing for web scraping
- **Readabilipy**: Content extraction and simplification
- **Markdownify**: HTML to Markdown conversion
- **Statistical Analysis**: Confidence scoring algorithms

## 🧠 Core Features

### 1. Advanced Personality Assessment

#### MBTI Quiz Engine
- **16-Question Assessment**: Scientifically-based questions covering all four personality axes
- **Multi-format Input Support**: 
  - Compact string format (`1a 2b 3c 4d`)
  - Structured JSON responses
  - Natural language answers ("strongly agree", "neutral")
- **Statistical Confidence Scoring**: Axis-by-axis confidence metrics (0-1 scale)
- **Response Sanitization**: Robust input validation and normalization

```python
# Example quiz generation
{
  "version": "1.0",
  "variant": "fixed", 
  "scale": {
    "labels": ["Strongly disagree", "Disagree", "Slightly disagree", "Neutral", 
               "Slightly agree", "Agree", "Strongly agree"],
    "values": [-3, -2, -1, 0, 1, 2, 3]
  },
  "questions": [
    {
      "id": "EI-1",
      "axis": "EI", 
      "prompt": "I feel energized by group conversations.",
      "positive_pole": "E"
    }
  ]
}
```

#### Personality Type Computation
- **Axis Scoring**: Independent scoring for E/I, S/N, T/F, J/P dimensions
- **Type Derivation**: Algorithmic computation of 16-type classification
- **Confidence Metrics**: Statistical reliability measures per axis
- **Data Validation**: Comprehensive input sanitization and error handling

### 2. Intelligent Matching System(Upcoming feature)

#### Compatibility Algorithm
- **Type Compatibility Scoring**: Multi-dimensional personality fit analysis
- **Shared Interest Detection**: Topic-based affinity matching
- **Availability Overlap**: Time-window intersection calculation
- **Intent Alignment**: Purpose-driven matching (networking, mentoring, etc.)

#### Matching Features
- **Ranked Results**: Scored candidate list with explanations
- **Plain-language Rationales**: Human-readable matching explanations
- **Configurable Limits**: Adjustable result set sizes
- **Real-time Updates**: Dynamic matching as profiles change

```python
# Example match result
{
  "match_id": "user1|user2",
  "score": 3.2,
  "rationale": "Matched due to shared interests in AI, startups, type fit INTJ × ENFP, 
               similar intent, and time overlap Monday 19:00-20:00.",
  "shared_topics": ["ai", "startups"],
  "availability_overlap": "Monday 19:00-20:00"
}
```

### 3. Comprehensive Coaching System

#### Communication Coaching
- **Type-specific Tips**: Tailored advice for communicating with each MBTI type
- **Context-aware Guidance**: Situation-specific coaching (feedback, collaboration, conflict)
- **Micro-coaching**: Quick, actionable suggestions for immediate use

#### Message Translation
- **Tone Adaptation**: Rewrite messages for specific personality types
- **Style Scaffolding**: Structured templates for effective communication
- **Preference Matching**: Adjust communication style to recipient preferences

#### Career & Relationship Guidance
- **Comprehensive Advice**: Detailed guidance for all 16 personality types
- **Strengths & Pitfalls**: Balanced perspective on type-specific challenges
- **Warning Signs**: Early indicators of potential issues
- **Success Strategies**: Proven approaches for each type

### 4. User Profile Management

#### Profile Components
- **Personality Data**: Type, confidence scores, quiz history
- **Availability Windows**: Flexible time-based scheduling
- **Interest Topics**: Tagged preference system
- **Intent Classification**: Purpose-driven categorization
- **Counterpart Management**: Stored reference types for coaching

#### Data Persistence
- **PostgreSQL Backend**: Reliable, ACID-compliant storage
- **Automatic Timestamps**: Created/updated tracking
- **JSONB Flexibility**: Schema-less preference storage
- **Migration Support**: Backward-compatible schema updates

## 🏗 Architecture

### Server Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    MCP Server (FastMCP)                    │
├─────────────────────────────────────────────────────────────┤
│  Authentication Layer (Bearer Token + RSA)                 │
├─────────────────────────────────────────────────────────────┤
│                    Business Logic                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐         │
│  │ Quiz Engine │ │  Matching   │ │  Coaching   │         │
│  │             │ │  Algorithm  │ │  System     │         │
│  └─────────────┘ └─────────────┘ └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│                 Database Layer (db.py)                     │
├─────────────────────────────────────────────────────────────┤
│               Supabase/PostgreSQL                          │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow
1. **Authentication**: Bearer token validation via RSA keypair
2. **Request Processing**: Pydantic validation and type checking
3. **Business Logic**: Domain-specific processing (quiz, matching, coaching)
4. **Data Persistence**: Async database operations via PostgREST
5. **Response Generation**: JSON serialization with error handling

### Database Schema
```sql
-- Core personality data
users_quiz (user_id, type, confidence_by_axis, axis_sums, raw_answers, sanitized_answers)

-- User preferences and matching data  
users_profile (user_id, type, availability, topics, intent, is_looking)

-- Coaching relationships
user_counterpart_type (user_id, counterpart_type)

-- Communication rooms
rooms (room_id, participants, tokens, expires_at)
```

## 🚀 Getting Started

### Prerequisites
- **Python 3.11+**: Modern Python with async support
- **Docker & Docker Compose**: For containerized deployment
- **Supabase Account**: For persistent data storage
- **ngrok Account**: For public HTTPS endpoint (development)

### Quick Setup

1. **Clone and Install**
```bash
git clone <repository-url>
cd puchaihackathon/mcp-starter
uv venv && uv sync
source .venv/bin/activate
```

2. **Configure Environment**
```bash
cp .env.example .env
# Edit .env with your tokens and Supabase credentials
```

3. **Set up Database**
```bash
# In Supabase SQL editor, run the contents of supabase_schema.sql
```

4. **Run Server**
```bash
# Local development
cd mcp-bearer-token && python mcp_starter.py

# Or with Docker
docker compose up -d
```

5. **Expose Publicly**
```bash
ngrok http 8086
# Note the HTTPS URL for Puch AI connection
```

### Environment Configuration

| Variable | Description | Required |
|----------|-------------|----------|
| `AUTH_TOKEN` | Secret authentication token | ✅ |
| `MY_NUMBER` | WhatsApp number (format: 919876543210) | ✅ |
| `SUPABASE_URL` | Supabase project URL | ✅ |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key | ✅ |
| `LOG_LEVEL` | Logging verbosity (INFO, DEBUG) | ❌ |
| `HTTP_MAX_CONNECTIONS` | HTTP client connection pool size | ❌ |
| `HTTP_MAX_KEEPALIVE` | HTTP keep-alive connections | ❌ |

## 🔧 API Reference

### Core Tools

#### `generate_quiz()`
Generates a standardized 16-question MBTI assessment.

**Parameters**: None (fixed configuration)
**Returns**: JSON quiz structure with questions and scale

#### `submit_quiz_compact(user_id, answers_compact)`
Processes quiz responses in compact string format.

**Parameters**:
- `user_id`: Unique user identifier
- `answers_compact`: String like "1a 2b 3c 4d 5e 6f 7g 8a..."

**Returns**: `{type: "INTJ", confidence_by_axis: {EI: 0.8, SN: 0.9, TF: 0.7, JP: 0.85}}`

#### `save_profile(user_id, type?, preferences?)`
Stores user profile and matching preferences.

**Parameters**:
- `user_id`: User identifier
- `type`: MBTI type (optional if quiz completed)
- `preferences`: Availability, topics, intent, is_looking

**Returns**: `{ok: true, type: "INTJ"}`

#### `find_matches(user_id, limit?)`
Finds compatible personality matches for a user.

**Parameters**:
- `user_id`: User to find matches for
- `limit`: Maximum results (default: 5)

**Returns**: Array of match objects with scores and rationales

#### `coach_tip(user_id, context, target_type?)`
Provides communication coaching for specific scenarios.

**Parameters**:
- `user_id`: Requesting user
- `context`: Situation description
- `target_type`: MBTI type to communicate with

**Returns**: `{target_type: "ESFJ", tips: ["tip1", "tip2", "tip3"]}`

#### `translate(message, target_type)`
Rewrites message for specific personality type.

**Parameters**:
- `message`: Original message
- `target_type`: MBTI type to adapt for

**Returns**: `{target_type: "ISFJ", rewritten: "adapted message"}`

### Utility Tools

#### `validate()`
Authentication validation (required by Puch AI).
**Returns**: Phone number string

#### `check_user_data_status(user_id)`
Checks existing user data to guide conversation flow.
**Returns**: Status object with data availability flags

#### `get_personality_guidance(user_id, guidance_type?)`
Comprehensive career and relationship advice.
**Returns**: Detailed guidance based on personality type

## 🎯 Usage Examples

### Complete Personality Assessment Flow

```bash
# 1. Generate quiz
/mcp tool generate_quiz {}

# 2. Present questions to user and collect responses
# 3. Submit compact answers
/mcp tool submit_quiz_compact {
  "user_id": "user123",
  "answers_compact": "1a 2c 3e 4b 5f 6d 7g 8a 9c 10e 11b 12f 13d 14g 15a 16c"
}

# 4. Save user preferences
/mcp tool save_profile {
  "user_id": "user123",
  "preferences": {
    "availability": [{"day": "monday", "start": "18:00", "end": "20:00"}],
    "topics": ["technology", "entrepreneurship"],
    "intent": "networking"
  }
}

# 5. Find matches
/mcp tool find_matches {"user_id": "user123", "limit": 3}
```

### Communication Coaching Example

```bash
# Set a counterpart type for ongoing coaching
/mcp tool set_counterpart {
  "user_id": "user123", 
  "counterpart_type": "ESFJ"
}

# Get situational coaching tips
/mcp tool coach_tip {
  "user_id": "user123",
  "context": "need to give constructive feedback about missed deadline"
}

# Translate message for specific type
/mcp tool translate {
  "message": "We need to discuss the project timeline",
  "target_type": "ISFP"
}
```

## 🔍 Technical Deep Dive

### Personality Assessment Algorithm

The assessment uses a sophisticated scoring system:

1. **Question Bank**: Carefully crafted questions for each MBTI axis
2. **Response Mapping**: Flexible input handling (letters, numbers, text)
3. **Confidence Calculation**: Statistical reliability per axis
4. **Type Derivation**: Algorithmic type computation from axis scores

### Matching Algorithm

Compatibility scoring considers multiple factors:

```python
def _type_fit_score(t1: str, t2: str, c1: dict, c2: dict) -> float:
    # Base compatibility by axis similarity/difference
    # Weighted by confidence levels
    # Returns 0-4 scale score
```

Factors:
- **Personality Compatibility**: MBTI type interaction patterns
- **Shared Interests**: Topic overlap analysis  
- **Schedule Alignment**: Time window intersection
- **Intent Matching**: Purpose compatibility

### Data Security & Privacy

- **Bearer Token Authentication**: Secure API access
- **Environment-based Secrets**: No hardcoded credentials
- **Input Sanitization**: Comprehensive validation
- **Error Handling**: Graceful failure modes
- **Audit Trails**: Automatic timestamp tracking

## 📊 Performance & Scalability

### Optimization Features
- **Connection Pooling**: Async HTTP client with persistent connections
- **Database Indexing**: Optimized queries with proper indexes
- **Caching Strategy**: In-memory caching for frequently accessed data
- **Async Architecture**: Non-blocking I/O throughout

### Monitoring & Logging
- **Structured Logging**: JSON-formatted logs with correlation IDs
- **Error Tracking**: Comprehensive exception handling
- **Performance Metrics**: Request timing and throughput monitoring
- **Health Checks**: Built-in endpoint monitoring

## 🚀 Deployment Options

### Development (Local)
```bash
python mcp-bearer-token/mcp_starter.py
ngrok http 8086
```

### Production (Docker)
```bash
docker compose up -d
# Configure reverse proxy (nginx, traefik) for HTTPS
```

### Cloud Platforms
- **Railway**: `railway up` with Dockerfile
- **Render**: Connect GitHub repo with auto-deploy
- **Heroku**: `git push heroku main`
- **DigitalOcean App Platform**: Import from GitHub

## 🤝 Connecting to Puch AI

1. **Start your MCP server** (local or deployed)
2. **Expose via HTTPS** (ngrok for development)
3. **Connect in Puch AI**:
   ```
   /mcp connect https://your-domain.ngrok.app/mcp your_auth_token
   ```
4. **Verify connection**:
   ```
   /mcp validate
   ```

## 🛡 Security Considerations

- **Authentication**: Bearer token with RSA signature validation
- **Input Validation**: Pydantic models with type checking
- **SQL Injection**: Parameterized queries via PostgREST
- **Rate Limiting**: HTTP client timeout and retry logic
- **Error Sanitization**: No sensitive data in error messages

## 🧪 Testing

### Unit Tests
```bash
pytest mcp-bearer-token/test_quiz_utils.py -v
```

### Integration Tests
```bash
# Test full quiz flow
/mcp tool generate_quiz {}
/mcp tool submit_quiz_compact {"user_id": "test", "answers_compact": "1a 2b 3c 4d"}
```

### Load Testing
```bash
# Test concurrent connections
ab -n 100 -c 10 http://localhost:8086/health
```

## 📈 Future Enhancements

### Planned Features
- **Advanced Analytics**: Personality trend analysis
- **Group Dynamics**: Team compatibility assessment  
- **Learning Algorithms**: Adaptive matching improvement
- **Mobile App**: Native iOS/Android clients
- **Video Coaching**: Real-time conversation analysis

### Potential Integrations
- **Calendar APIs**: Automatic availability sync
- **Slack/Teams**: Workplace personality insights
- **CRM Systems**: Sales personality matching
- **HR Platforms**: Recruitment and team building

## 📚 Resources & Documentation

### Official Documentation
- **Puch AI MCP Docs**: https://puch.ai/mcp
- **FastMCP Framework**: https://github.com/puchao/fastmcp
- **Model Context Protocol**: https://modelcontextprotocol.io

### MBTI & Personality Science
- **Myers-Briggs Foundation**: https://www.myersbriggs.org
- **Personality Research**: Academic papers on type theory
- **Statistical Validation**: Reliability and validity studies

### Technical References
- **Async Python**: https://docs.python.org/3/library/asyncio.html
- **PostgreSQL JSONB**: https://www.postgresql.org/docs/current/datatype-json.html
- **Supabase**: https://supabase.com/docs

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Setup
```bash
git clone <your-fork>
cd puchaihackathon/mcp-starter
uv venv && uv sync
pre-commit install  # Code formatting and linting
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Hackathon Context

This project was built for the **Puch AI Hackathon** to demonstrate the capabilities of the Model Context Protocol in creating sophisticated AI-powered applications. It showcases:

- **Advanced MCP Integration**: Complex tool orchestration
- **Real-world Application**: Practical personality coaching
- **Technical Excellence**: Production-ready architecture
- **Innovation**: Novel personality assessment approach

## 📞 Support & Contact

- **Puch AI Discord**: https://discord.gg/VMCnMvYx
- **Puch AI Documentation**: https://puch.ai/mcp
- **Project Issues**: [GitHub Issues](issues)
- **Puch WhatsApp**: +91 99988 81729

---

**Built with ❤️ for the Puch AI Hackathon**

*Use hashtag `#BuildWithPuch` when sharing this project!*
