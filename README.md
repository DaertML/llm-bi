# llm-bi
Business Intelligence and Data Querying that uses natural language processing with LLMs, so that you can forget about SQL

# Introduction
Just running "docker-compose up" should make it; then add the data to the buckets. If you want to do it properly, customize the docker-compose Minio's commands to create the buckets that you need and copy the data from there, or do it with the Web UI.

The agent needs more work, as the harness is making the agent just give summaries instead of raw data when asked for it. Somehow, it is a working PoC.

# Future Work
Bring true BI with plots and diagrams that can be created on the go.

# Credits
The idea of this and architecture are from Pablo Dafonte Iglesias; who seeded this idea in me a long time ago, somehow the lack of expertise with this and testing of different frameworks didnt bring it to fruition; for which a quick demo he showcased me got the pieces together to quickly prototype this with Claude. Also, thanks to MotherDuck, as this has been an inspiration: https://www.youtube.com/watch?v=YnqcJYAQnzQ&t=513s
