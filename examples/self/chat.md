---
---
PREFIX ex:   <http://example.org/> .
PREFIX mcp:  <http://modelcontextprotocol.org/> .
PREFIX schema: <http://schema.org/> .

# -----------------------------------------------------------------
# 1.  TYPE declarations – the core vocabulary used in the conversation
# -----------------------------------------------------------------
TYPE ex:Service .
TYPE ex:Provider .
TYPE ex:Protocol .
TYPE ex:Capability .
TYPE ex:Operation .
TYPE ex:Plan .
TYPE ex:Decision .
TYPE ex:Event .
TYPE ex:Connection .
TYPE ex:Credential .

# -----------------------------------------------------------------
# 2.  CAPABILITY declarations – the canonical MCP operations that were
#     agreed upon in the discussion
# -----------------------------------------------------------------
CAPABILITY mcp:calendar.search ( query: schema:Text OPTIONAL ) RETURNS ex:Operation .
CAPABILITY mcp:calendar.get    ( id: schema:Text ONE )      RETURNS ex:Operation .
CAPABILITY mcp:calendar.create( event: ex:CalendarEvent ONE ) RETURNS ex:Operation .
CAPABILITY mcp:mail.send      ( message: ex:MailMessage ONE ) RETURNS ex:Operation .
CAPABILITY mcp:storage.read   ( objectId: schema:Text ONE )  RETURNS ex:Operation .
CAPABILITY mcp:contacts.search( query: schema:Text OPTIONAL ) RETURNS ex:Operation .

# -----------------------------------------------------------------
# 3.  EVENT schema – what an “event” looks like in the system
# -----------------------------------------------------------------
EVENT ex:DecisionMade
    ( decisionId: schema:Text
      description: schema:Text
      madeBy:     schema:Person
      timestamp:  schema:DateTime ) .

# -----------------------------------------------------------------
# 4.  WORLD graph – the factual statements that capture the design
# -----------------------------------------------------------------
WORLD {
    # 4.1  High‑level purpose
    ex:ProductivitySuite ex:hasPurpose
        "Provide a unified, provider‑neutral MCP control plane for calendars, mail, storage, contacts and documents." .

    # 4.2  Core principles
    ex:ProductivitySuite ex:corePrinciple "Provider neutrality – use canonical domain models only." .
    ex:ProductivitySuite ex:corePrinciple "Capability‑oriented design – operations are first‑class." .
    ex:ProductivitySuite ex:corePrinciple "Policy boundary before any external call." .
    ex:ProductivitySuite ex:corePrinciple "Audit every operation." .

    # 4.3  Services that must exist
    ex:Calendar   a ex:Service .
    ex:Mail       a ex:Service .
    ex:Storage    a ex:Service .
    ex:Contacts   a ex:Service .

    # 4.4  Providers and the protocols they expose
    ex:Google    a ex:Provider ;
                 ex:supports ex:Calendar , ex:Mail , ex:Storage .
    ex:Microsoft a ex:Provider ;
                 ex:supports ex:Calendar , ex:Mail , ex:Storage .
    ex:Apple     a ex:Provider ;
                 ex:supports ex:Calendar , ex:Mail .
    ex:Dropbox   a ex:Provider ;
                 ex:supports ex:Storage .
    ex:AWS_S3    a ex:Provider ;
                 ex:supports ex:Storage .

    ex:Google   ex:usesProtocol ex:GoogleREST .
    ex:Microsoft ex:usesProtocol ex:MicrosoftGraph .
    ex:Apple    ex:usesProtocol ex:CalDAV , ex:IMAP , ex:SMTP , ex:CardDAV .
    ex:Dropbox  ex:usesProtocol ex:DropboxAPI .
    ex:AWS_S3   ex:usesProtocol ex:S3API .

    # 4.5  Canonical models (only names – details are in the spec text)
    ex:CalendarEvent   a ex:Capability .
    ex:MailMessage     a ex:Capability .
    ex:StorageObject   a ex:Capability .
    ex:Contact         a ex:Capability .

    # 4.6  Decisions that were taken (each emitted as an event)
    ex:Decision1 a ex:Decision ;
        ex:hasId "DEC-001" ;
        ex:hasDescription "Make providers independent npm packages" ;
        ex:madeBy "User" ;
        ex:timestamp "2024-09-11T12:00:00Z" .

    ex:Decision2 a ex:Decision ;
        ex:hasId "DEC-002" ;
        ex:hasDescription "Treat Apple as protocol adapters, not a single AppleAdapter" ;
        ex:madeBy "Assistant" ;
        ex:timestamp "2024-09-11T12:01:15Z" .

    ex:Decision3 a ex:Decision ;
        ex:hasId "DEC-003" ;
        ex:hasDescription "Do not expose SDK types outside adapters" ;
        ex:madeBy "User" ;
        ex:timestamp "2024-09-11T12:02:30Z" .
} .

# -----------------------------------------------------------------
# 5.  PLAN – a high‑level execution plan that encodes the agreed steps
# -----------------------------------------------------------------
PLAN ex:DefineMcpSpec ( ) {
    # 5.1  Record the purpose and core principles
    WRITE WORLD {
        ex:ProductivitySuite ex:hasStatus "SpecificationInProgress" .
    } .

    # 5.2  Emit decision events so that the audit log knows what was decided
    EMIT ex:DecisionMade(
        decisionId   "DEC-001" ,
        description  "Make providers independent npm packages" ,
        madeBy       "User" ,
        timestamp    "2024-09-11T12:00:00Z"
    ) .

    EMIT ex:DecisionMade(
        decisionId   "DEC-002" ,
        description  "Treat Apple as protocol adapters, not a single AppleAdapter" ,
        madeBy       "Assistant" ,
        timestamp    "2024-09-11T12:01:15Z"
    ) .

    EMIT ex:DecisionMade(
        decisionId   "DEC-003" ,
        description  "Do not expose SDK types outside adapters" ,
        madeBy       "User" ,
        timestamp    "2024-09-11T12:02:30Z"
    ) .

    # 5.3  Declare the concrete provider packages (metadata only)
    WRITE WORLD {
        ex:PackageCore          a ex:Package ;
            ex:dependsOn " @modelcontextprotocol/sdk , zod , jose , openid-client , pino " .

        ex:PackageProviderGoogle   a ex:Package ;
            ex:dependsOn " googleapis " .

        ex:PackageProviderMicrosoft a ex:Package ;
            ex:dependsOn " @microsoft/microsoft-graph-client , @azure/identity " .

        ex:PackageProviderApple    a ex:Package ;
            ex:dependsOn " tsdav " .

        ex:PackageProviderDropbox  a ex:Package ;
            ex:dependsOn " dropbox " .

        ex:PackageProviderS3       a ex:Package ;
            ex:dependsOn " @aws-sdk/client-s3 , @aws-sdk/lib-storage " .
    } .

    # 5.4  Mark the specification as complete
    WRITE WORLD {
        ex:ProductivitySuite ex:hasStatus "SpecificationComplete" .
    } .

    RETURN ex:ProductivitySuite .
} .

# -----------------------------------------------------------------
# 6.  OPTIONAL: a small proof‑of‑concept MATCH that extracts all
#    decisions from the WORLD graph and returns them
# -----------------------------------------------------------------
MATCH WORLD {
    ?d a ex:Decision ;
       ex:hasId ?id ;
       ex:hasDescription ?desc ;
       ex:madeBy ?author ;
       ex:timestamp ?ts .
} AS my:?decisions .

RETURN my:?decisions .