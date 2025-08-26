#!/usr/bin/env bash
curl -s -X POST "http://localhost:8000/answer"       -H "Content-Type: application/json"       -d '{"question":"How do I transfer a 401k after closing a business?","top_k":12,"allow_external_knowledge":true}' | jq
