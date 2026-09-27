# Cloud Deployment Guide

This directory contains configuration files for deploying the Cerebro-X Brain Twin to various cloud providers.

## Heroku
1. Install the Heroku CLI and login: `heroku login`
2. Create a new app: `heroku create cerebro-x`
3. Set the stack to container: `heroku stack:set container`
4. Add the Postgres add-on (optional, if moving off SQLite): `heroku addons:create heroku-postgresql:mini`
5. Push the container: `git push heroku main`

## AWS Elastic Beanstalk
1. Install the EB CLI: `pip install awsebcli`
2. Initialize the application: `eb init -p docker cerebro-x`
3. Create the environment: `eb create cerebro-x-env`
4. Deploy the latest version: `eb deploy`

## Environment Variables required for Production
- `FASTAPI_ENV=production`
- `DATABASE_URL` (If you are moving away from the default SQLite file).

**Note:** Be sure to configure CORS appropriately in `main.py` when deploying to a public domain.
