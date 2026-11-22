# -------------------------------------------------------------
# 1. PREFIXES
# -------------------------------------------------------------
PREFIX mcp:   <https://modelcontextprotocol.org/mcp#> .
PREFIX prov:  <https://modelcontextprotocol.org/provider#> .
PREFIX schema: <http://schema.org/> .
PREFIX xsd:   <http://www.w3.org/2001/XMLSchema#> .
PREFIX ex:    <https://example.com/vocab#> .

# -------------------------------------------------------------
# 2. TYPE DECLARATIONS (canonical domain models)
# -------------------------------------------------------------
TYPE mcp:CalendarEvent .
TYPE mcp:MailMessage .
TYPE mcp:StorageObject .
TYPE mcp:Contact .
TYPE mcp:Connection .
TYPE mcp:Credential .
TYPE mcp:Operation .
TYPE mcp:Policy .
TYPE mcp:AuditEvent .
TYPE mcp:Error .

# -------------------------------------------------------------
# 3. SERVICE PORTS (abstract capability interfaces)
# -------------------------------------------------------------
TYPE mcp:CalendarPort   EXTENDS mcp:Port .
TYPE mcp:MailPort       EXTENDS mcp:Port .
TYPE mcp:StoragePort    EXTENDS mcp:Port .
TYPE mcp:ContactsPort   EXTENDS mcp:Port .

# -------------------------------------------------------------
# 4. PROVIDER ADAPTERS (concrete implementations of ports)
# -------------------------------------------------------------
TYPE prov:GoogleCalendarAdapter   EXTENDS mcp:CalendarPort .
TYPE prov:MicrosoftCalendarAdapter EXTENDS mcp:CalendarPort .
TYPE prov:CalDAVCalendarAdapter    EXTENDS mcp:CalendarPort .

TYPE prov:GmailAdapter            EXTENDS mcp:MailPort .
TYPE prov:MicrosoftGraphMailAdapter EXTENDS mcp:MailPort .
TYPE prov:ImapSmtpMailAdapter     EXTENDS mcp:MailPort .

TYPE prov:GoogleDriveAdapter      EXTENDS mcp:StoragePort .
TYPE prov:DropboxAdapter          EXTENDS mcp:StoragePort .
TYPE prov:S3Adapter               EXTENDS mcp:StoragePort .
TYPE prov:FilesystemAdapter       EXTENDS mcp:StoragePort .

# -------------------------------------------------------------
# 5. CAPABILITY DECLARATIONS (MCP operations)
# -------------------------------------------------------------
CAPABILITY mcp:calendar.search
    ( query : mcp:CalendarQuery OPTIONAL,
      pagination : mcp:PageRequest OPTIONAL )
    RETURNS mcp:CalendarSearchResult .

CAPABILITY mcp:calendar.get
    ( eventId : xsd:string ONE )
    RETURNS mcp:CalendarEvent .

CAPABILITY mcp:calendar.create
    ( event : mcp:CalendarEvent ONE )
    RETURNS mcp:CalendarEvent .

CAPABILITY mcp:calendar.update
    ( eventId : xsd:string ONE,
      patch   : mcp:CalendarEventPatch ONE )
    RETURNS mcp:CalendarEvent .

CAPABILITY mcp:calendar.delete
    ( eventId : xsd:string ONE )
    RETURNS xsd:boolean .

CAPABILITY mcp:mail.search
    ( query : mcp:MailQuery OPTIONAL,
      pagination : mcp:PageRequest OPTIONAL )
    RETURNS mcp:MailSearchResult .

CAPABILITY mcp:mail.get
    ( messageId : xsd:string ONE )
    RETURNS mcp:MailMessage .

CAPABILITY mcp:mail.send
    ( message : mcp:MailMessage ONE,
      idempotencyKey : xsd:string OPTIONAL )
    RETURNS mcp:MailMessage .

CAPABILITY mcp:mail.reply
    ( threadId : xsd:string ONE,
      reply    : mcp:MailMessage ONE,
      idempotencyKey : xsd:string OPTIONAL )
    RETURNS mcp:MailMessage .

CAPABILITY mcp:storage.list
    ( prefix : xsd:string OPTIONAL,
      pagination : mcp:PageRequest OPTIONAL )
    RETURNS mcp:StorageListResult .

CAPABILITY mcp:storage.read
    ( objectId : xsd:string ONE )
    RETURNS mcp:ReadableStream .

CAPABILITY mcp:storage.write
    ( objectId : xsd:string ONE,
      content   : mcp:ReadableStream ONE,
      metadata  : mcp:Metadata OPTIONAL,
      idempotencyKey : xsd:string OPTIONAL )
    RETURNS xsd:boolean .

CAPABILITY mcp:contacts.search
    ( query : mcp:ContactQuery OPTIONAL,
      pagination : mcp:PageRequest OPTIONAL )
    RETURNS mcp:ContactSearchResult .

CAPABILITY mcp:contacts.get
    ( contactId : xsd:string ONE )
    RETURNS mcp:Contact .

# -------------------------------------------------------------
# 6. CONNECTION & CREDENTIAL MODELS
# -------------------------------------------------------------
TYPE mcp:ConnectionState ENUM {
    connected,
    expired,
    revoked,
    unavailable
} .

TYPE mcp:CredentialType ENUM {
    oauth2,
    oidc,
    apiKey,
    iam,
    password
} .

# -------------------------------------------------------------
# 7. POLICY & AUDIT MODELS
# -------------------------------------------------------------
TYPE mcp:Effect ENUM { read, write, external_communication, destructive } .
TYPE mcp:Sensitivity ENUM { low, normal, high, critical } .
TYPE mcp:Confirmation ENUM { never, conditional, always } .

# -------------------------------------------------------------
# 8. ERROR HIERARCHY
# -------------------------------------------------------------
TYPE mcp:AuthenticationError   EXTENDS mcp:Error .
TYPE mcp:AuthorizationError    EXTENDS mcp:Error .
TYPE mcp:NotFoundError          EXTENDS mcp:Error .
TYPE mcp:ConflictError          EXTENDS mcp:Error .
TYPE mcp:RateLimitError         EXTENDS mcp:Error .
TYPE mcp:ValidationError        EXTENDS mcp:Error .
TYPE mcp:UnavailableError       EXTENDS mcp:Error .
TYPE mcp:UnsupportedOperationError EXTENDS mcp:Error .
TYPE mcp:ProviderError          EXTENDS mcp:Error .

# -------------------------------------------------------------
# 9. GRAPH CONTEXTS – DECLARE THE “WORLD” (runtime data)
# -------------------------------------------------------------
WORLD {
    # Providers advertised in the catalogue
    prov:google   a prov:Provider ;
    prov:microsoft a prov:Provider ;
    prov:apple    a prov:Provider ;
    prov:dropbox  a prov:Provider ;
    prov:aws      a prov:Provider .

    # Service‑to‑port mapping (canonical)
    mcp:calendar   mcp:hasPort  mcp:CalendarPort .
    mcp:mail       mcp:hasPort  mcp:MailPort .
    mcp:storage    mcp:hasPort  mcp:StoragePort .
    mcp:contacts   mcp:hasPort  mcp:ContactsPort .

    # Provider‑adapter wiring
    prov:google   mcp:providesAdapter prov:GoogleCalendarAdapter .
    prov:google   mcp:providesAdapter prov:GmailAdapter .
    prov:google   mcp:providesAdapter prov:GoogleDriveAdapter .

    prov:microsoft mcp:providesAdapter prov:MicrosoftCalendarAdapter .
    prov:microsoft mcp:providesAdapter prov:MicrosoftGraphMailAdapter .
    prov:microsoft mcp:providesAdapter prov:OneDriveAdapter .

    prov:apple    mcp:providesAdapter prov:CalDAVCalendarAdapter .
    prov:apple    mcp:providesAdapter prov:ImapSmtpMailAdapter .

    prov:dropbox  mcp:providesAdapter prov:DropboxAdapter .
    prov:aws      mcp:providesAdapter prov:S3Adapter .
} .

# -------------------------------------------------------------
# 10. SAMPLE CONNECTION INSTANCES (runtime configuration)
# -------------------------------------------------------------
MEMORY {
    # Google personal account
    ex:conn_google_personal a mcp:Connection ;
        mcp:service    "calendar" ;
        mcp:provider   "google" ;
        mcp:protocol   "google-calendar" ;
        mcp:accountId  "105678901234567890123" ;
        mcp:state      mcp:connected .

    # Microsoft corporate account (Graph)
    ex:conn_ms_graph a mcp:Connection ;
        mcp:service    "mail" ;
        mcp:provider   "microsoft" ;
        mcp:protocol   "microsoft-graph" ;
        mcp:accountId  "c1d2e3f4-5678-90ab-cdef-1234567890ab" ;
        mcp:state      mcp:connected .

    # Apple iCloud CalDAV connection
    ex:conn_apple_caldav a mcp:Connection ;
        mcp:service    "calendar" ;
        mcp:provider   "apple" ;
        mcp:protocol   "caldav" ;
        mcp:accountId  "icloud-user@example.com" ;
        mcp:state      mcp:connected .
} .

# -------------------------------------------------------------
# 11. OPERATION DESCRIPTORS (MCP tools – projection)
# -------------------------------------------------------------
PROJECTION calendar.search ( query : mcp:CalendarQuery OPTIONAL ) {
    # The projection simply forwards to the capability
    CALL mcp:calendar.search ( query ) AS my:?result .
    RETURN my:?result .
} .

PROJECTION mail.send ( message : mcp:MailMessage ONE,
                      idempotencyKey : xsd:string OPTIONAL ) {
    CALL mcp:mail.send ( message , idempotencyKey ) AS my:?sent .
    RETURN my:?sent .
} .

PROJECTION storage.write ( objectId : xsd:string ONE,
                          content  : mcp:ReadableStream ONE,
                          metadata : mcp:Metadata OPTIONAL,
                          idempotencyKey : xsd:string OPTIONAL ) {
    CALL mcp:storage.write ( objectId , content , metadata , idempotencyKey )
        AS my:?ok .
    RETURN my:?ok .
} .

# -------------------------------------------------------------
# 12. POLICY EXAMPLE (declarative rule)
# -------------------------------------------------------------
PLAN policy_enforce_mail_send (
    connectionId : xsd:string,
    operationId   : xsd:string
) {
    # Resolve connection & operation
    MATCH WORLD {
        my:?conn  a mcp:Connection ;
                mcp:id         my:?connectionId ;
                mcp:service    "mail" .
        my:?op    a mcp:Operation ;
                mcp:id         my:?operationId ;
                mcp:effects    ?effect .
    } AS my:?bindings .

    # Simple policy: any operation with external_communication effect
    # requires confirmation
    BRANCH {
        WHEN my:?effect = mcp:external_communication THEN {
            CALL mcp:requireConfirmation ( my:?operationId ) AS my:?confirmed .
            BRANCH {
                WHEN my:?confirmed = true THEN {
                    RETURN "allowed" .
                }
                ELSE {
                    RETURN "denied" .
                }
            }
        }
        ELSE {
            RETURN "allowed" .
        }
    } .
} .

# -------------------------------------------------------------
# 13. AUDIT EVENT EMISSION (after each operation)
# -------------------------------------------------------------
EMIT ex:AuditEvent(
    id:          "audit-" || UUID(),
    timestamp:   NOW(),
    actor:       my:?actorId,
    operation:   my:?opId,
    service:     my:?service,
    connection:  my:?connId,
    decision:    my:?decision,
    outcome:     my:?outcome,
    durationMs:  my:?duration
) .

# -------------------------------------------------------------
# 14. MVP PLAN (orchestrating a simple end‑to‑end flow)
# -------------------------------------------------------------
PLAN mvp_calendar_event_creation (
    connectionId : xsd:string,
    calendarId   : xsd:string,
    title        : xsd:string,
    start        : xsd:dateTime,
    end          : xsd:dateTime
) {
    # 1️⃣ Resolve connection & ensure it’s a Calendar provider
    MATCH WORLD {
        my:?conn a mcp:Connection ;
                mcp:id   my:?connectionId ;
                mcp:service "calendar" .
    } AS my:?connBinding .

    # 2️⃣ Build the canonical event
    BIND my:?event = {
        mcp:id        UUID() ;
        mcp:calendarId my:?calendarId ;
        mcp:title     my:?title ;
        mcp:start     my:?start ;
        mcp:end       my:?end ;
        mcp:timezone  "UTC"
    } .

    # 3️⃣ Call the create capability (writes to external provider)
    CALL mcp:calendar.create ( my:?event ) AS my:?created .

    # 4️⃣ Emit an audit record
    EMIT ex:AuditEvent(
        id:        "audit-" || UUID(),
        timestamp: NOW(),
        actor:     "system",
        operation: "calendar.create",
        service:   "calendar",
        connection: my:?connectionId,
        decision:  "allowed",
        outcome:   "success",
        durationMs: 0
    ) .

    # 5️⃣ Return the created event
    RETURN my:?created .
} .

# -------------------------------------------------------------
# 15. END‑OF‑SPEC COMMENT
# -------------------------------------------------------------
# This Graph IR program fully describes the **Multi‑Service MCP Productivity Suite**:
# • Types for all canonical domain objects (calendar, mail, storage, contacts, …)
# • Abstract ports and concrete provider adapters
# • Capability (operation) declarations with input/output schemas
# • Connection and credential models (provider‑neutral)
# • Policy, audit, and error vocabularies
# • Sample projections (MCP tools), a generic policy plan, and an MVP orchestration plan
# • All data lives in the appropriate graph contexts (WORLD for static catalogue,
#   MEMORY for runtime configuration, PLAN for executable procedures).

# The program can be parsed, type‑checked, and normalised to the twelve primitive VM
# operations (MATCH, READ, WRITE, DELETE, BIND, DERIVE, CALL, EMIT, SPAWN, WAIT,
# BRANCH, RETURN) by any conforming Graph IR compiler.
