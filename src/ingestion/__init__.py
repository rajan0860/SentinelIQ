"""
Ingestion Package
=================
Handles the full data ingestion and processing pipeline.

EventLoader     → loads and validates events.csv
GraphBuilder    → builds the NetworkX account-device-IP graph
IngestionPipeline → orchestrates the full flow end-to-end
"""
