#!/bin/bash

# Docker Compose'u başlat
docker-compose up -d

# Eğer logları izlemek istersen (isteğe bağlı)
docker-compose logs -f
