
Generate tpcds queries:
```bash
./dsqgen \
    -DIRECTORY ../query_templates \
    -INPUT ../query_templates/templates.lst \
    -VERBOSE Y \
    -QUALIFY Y \
    -SCALE 1 \
    -DIALECT netezza \
    -OUTPUT_DIR /tmp \
    -STREAMS 10
```