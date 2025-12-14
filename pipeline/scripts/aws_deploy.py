#!/usr/bin/env python3
"""
============================================================================
AWS DEPLOYMENT AUTOMATION SCRIPT
============================================================================
Purpose: Automated deployment to AWS (ECR, ECS, CloudFormation, Alarms)
Usage: python scripts/aws_deploy.py [options]
Author: GATE AE SOTA Pipeline
============================================================================
"""

import os
import sys
import json
import boto3
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))


class AWSDeployer:
    """
    Automated AWS deployment for GATE AE Pipeline
    """
    
    def __init__(
        self,
        environment: str = "production",
        region: str = "ap-south-1",
        profile: Optional[str] = None
    ):
        self.environment = environment
        self.region = region
        self.project_name = "gate-ae-pipeline"
        
        # Create boto3 session
        session_kwargs = {"region_name": region}
        if profile:
            session_kwargs["profile_name"] = profile
        
        self.session = boto3.Session(**session_kwargs)
        
        # Initialize clients
        self.ecr = self.session.client('ecr')
        self.ecs = self.session.client('ecs')
        self.iam = self.session.client('iam')
        self.cloudformation = self.session.client('cloudformation')
        self.cloudwatch = self.session.client('cloudwatch')
        self.sns = self.session.client('sns')
        self.sts = self.session.client('sts')
        
        # Get account ID
        self.account_id = self.sts.get_caller_identity()['Account']
        
        print(f"🚀 AWS Deployer initialized")
        print(f"   Environment: {environment}")
        print(f"   Region: {region}")
        print(f"   Account: {self.account_id}")
        print()
    
    def deploy_all(self):
        """Run complete deployment"""
        print("=" * 80)
        print("COMPLETE AWS DEPLOYMENT")
        print("=" * 80)
        print()
        
        # Step 1: Create ECR repository
        self.create_ecr_repository()
        
        # Step 2: Build and push Docker image
        self.build_and_push_image()
        
        # Step 3: Deploy CloudFormation stack
        self.deploy_cloudformation()
        
        # Step 4: Setup CloudWatch alarms
        self.setup_cloudwatch_alarms()
        
        # Step 5: Register ECS task definition
        self.register_task_definition()
        
        print()
        print("=" * 80)
        print("✅ DEPLOYMENT COMPLETE")
        print("=" * 80)
    
    def create_ecr_repository(self):
        """Create ECR repository if it doesn't exist"""
        print("[1/5] Creating ECR repository...")
        
        repo_name = f"{self.project_name}-{self.environment}"
        
        try:
            # Check if exists
            self.ecr.describe_repositories(repositoryNames=[repo_name])
            print(f"   ✓ Repository already exists: {repo_name}")
        
        except self.ecr.exceptions.RepositoryNotFoundException:
            # Create repository
            response = self.ecr.create_repository(
                repositoryName=repo_name,
                imageScanningConfiguration={'scanOnPush': True},
                encryptionConfiguration={'encryptionType': 'AES256'},
                tags=[
                    {'Key': 'Environment', 'Value': self.environment},
                    {'Key': 'Project', 'Value': self.project_name}
                ]
            )
            
            repo_uri = response['repository']['repositoryUri']
            print(f"   ✓ Created repository: {repo_uri}")
        
        except Exception as e:
            print(f"   ✗ Error: {e}")
            raise
    
    def build_and_push_image(self):
        """Build Docker image and push to ECR"""
        print("\n[2/5] Building and pushing Docker image...")
        
        repo_name = f"{self.project_name}-{self.environment}"
        image_uri = f"{self.account_id}.dkr.ecr.{self.region}.amazonaws.com/{repo_name}"
        
        # Generate image tag
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        git_commit = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True
        ).stdout.strip() or 'unknown'
        
        image_tag = f"{timestamp}-{git_commit}"
        
        try:
            # ECR login
            print("   Logging into ECR...")
            login_cmd = subprocess.run(
                [
                    'aws', 'ecr', 'get-login-password',
                    '--region', self.region
                ],
                capture_output=True,
                text=True
            )
            
            subprocess.run(
                [
                    'docker', 'login',
                    '--username', 'AWS',
                    '--password-stdin',
                    f"{self.account_id}.dkr.ecr.{self.region}.amazonaws.com"
                ],
                input=login_cmd.stdout,
                text=True,
                check=True
            )
            
            # Build image
            print(f"   Building image: {image_tag}...")
            subprocess.run(
                [
                    'docker', 'build',
                    '-f', 'docker/Dockerfile',
                    '-t', f"{image_uri}:{image_tag}",
                    '-t', f"{image_uri}:latest",
                    '--build-arg', f"BUILD_DATE={datetime.utcnow().isoformat()}",
                    '--build-arg', f"VERSION={image_tag}",
                    '.'
                ],
                cwd=PROJECT_ROOT,
                check=True
            )
            
            # Push image
            print(f"   Pushing to ECR...")
            subprocess.run(['docker', 'push', f"{image_uri}:{image_tag}"], check=True)
            subprocess.run(['docker', 'push', f"{image_uri}:latest"], check=True)
            
            print(f"   ✓ Image pushed: {image_uri}:{image_tag}")
            
            # Store for later use
            self.image_uri = f"{image_uri}:{image_tag}"
        
        except subprocess.CalledProcessError as e:
            print(f"   ✗ Build/push failed: {e}")
            raise
    
    def deploy_cloudformation(self):
        """Deploy CloudFormation stack"""
        print("\n[3/5] Deploying CloudFormation stack...")
        
        stack_name = f"{self.project_name}-{self.environment}"
        template_path = PROJECT_ROOT / "aws" / "cloudformation_template.yaml"
        
        try:
            # Read template
            with open(template_path, 'r') as f:
                template_body = f.read()
            
            # Check if stack exists
            try:
                self.cloudformation.describe_stacks(StackName=stack_name)
                stack_exists = True
            except self.cloudformation.exceptions.ClientError:
                stack_exists = False
            
            # Parameters
            parameters = [
                {'ParameterKey': 'Environment', 'ParameterValue': self.environment},
                {'ParameterKey': 'ProjectName', 'ParameterValue': self.project_name},
            ]
            
            if stack_exists:
                # Update stack
                print(f"   Updating stack: {stack_name}...")
                self.cloudformation.update_stack(
                    StackName=stack_name,
                    TemplateBody=template_body,
                    Parameters=parameters,
                    Capabilities=['CAPABILITY_NAMED_IAM']
                )
                
                # Wait for update
                print("   Waiting for stack update...")
                waiter = self.cloudformation.get_waiter('stack_update_complete')
                waiter.wait(StackName=stack_name)
            
            else:
                # Create stack
                print(f"   Creating stack: {stack_name}...")
                self.cloudformation.create_stack(
                    StackName=stack_name,
                    TemplateBody=template_body,
                    Parameters=parameters,
                    Capabilities=['CAPABILITY_NAMED_IAM'],
                    Tags=[
                        {'Key': 'Environment', 'Value': self.environment},
                        {'Key': 'Project', 'Value': self.project_name}
                    ]
                )
                
                # Wait for creation
                print("   Waiting for stack creation...")
                waiter = self.cloudformation.get_waiter('stack_create_complete')
                waiter.wait(StackName=stack_name)
            
            # Get outputs
            response = self.cloudformation.describe_stacks(StackName=stack_name)
            outputs = response['Stacks'][0].get('Outputs', [])
            
            print(f"   ✓ Stack deployed successfully")
            print(f"   Stack outputs:")
            for output in outputs:
                print(f"     - {output['OutputKey']}: {output['OutputValue']}")
        
        except Exception as e:
            print(f"   ✗ Stack deployment failed: {e}")
            raise
    
    def setup_cloudwatch_alarms(self):
        """Setup CloudWatch alarms"""
        print("\n[4/5] Setting up CloudWatch alarms...")
        
        alarms_file = PROJECT_ROOT / "aws" / "cloudwatch_alarms.json"
        
        try:
            # Load alarms definition
            with open(alarms_file, 'r') as f:
                alarms_config = json.load(f)
            
            # Create SNS topics first
            sns_topics = alarms_config.get('sns_topics', {})
            topic_arns = {}
            
            for topic_name, topic_config in sns_topics.items():
                try:
                    response = self.sns.create_topic(Name=topic_name)
                    topic_arns[topic_name] = response['TopicArn']
                    print(f"   ✓ SNS topic: {topic_name}")
                except Exception as e:
                    print(f"   ⚠ Topic might exist: {topic_name}")
            
            # Create alarms
            created_count = 0
            for alarm in alarms_config.get('alarms', []):
                try:
                    metric = alarm['metric']
                    
                    # Replace ACCOUNT_ID placeholder in alarm actions
                    alarm_actions = []
                    for action in alarm['actions'].get('alarm_actions', []):
                        alarm_actions.append(
                            action.replace('ACCOUNT_ID', self.account_id)
                        )
                    
                    # Create alarm
                    self.cloudwatch.put_metric_alarm(
                        AlarmName=alarm['alarm_name'],
                        AlarmDescription=alarm['description'],
                        MetricName=metric['metric_name'],
                        Namespace=metric['namespace'],
                        Statistic=metric['statistic'],
                        Period=metric['period'],
                        EvaluationPeriods=metric['evaluation_periods'],
                        Threshold=metric['threshold'],
                        ComparisonOperator=metric['comparison_operator'],
                        TreatMissingData=metric.get('treat_missing_data', 'notBreaching'),
                        AlarmActions=alarm_actions
                    )
                    
                    created_count += 1
                
                except Exception as e:
                    print(f"   ⚠ Failed to create alarm {alarm['alarm_name']}: {e}")
            
            print(f"   ✓ Created {created_count} CloudWatch alarms")
        
        except Exception as e:
            print(f"   ✗ Alarm setup failed: {e}")
            raise
    
    def register_task_definition(self):
        """Register ECS task definition"""
        print("\n[5/5] Registering ECS task definition...")
        
        task_def_file = PROJECT_ROOT / "aws" / "ecs_task_definition.json"
        
        try:
            # Load task definition
            with open(task_def_file, 'r') as f:
                task_def = json.load(f)
            
            # Update with actual values
            task_def['family'] = f"{self.project_name}-{self.environment}"
            task_def['containerDefinitions'][0]['image'] = self.image_uri
            
            # Replace placeholders in ARNs
            execution_role = task_def['executionRoleArn'].replace('YOUR_ACCOUNT_ID', self.account_id)
            task_role = task_def['taskRoleArn'].replace('YOUR_ACCOUNT_ID', self.account_id)
            
            task_def['executionRoleArn'] = execution_role
            task_def['taskRoleArn'] = task_role
            
            # Register
            response = self.ecs.register_task_definition(**task_def)
            task_def_arn = response['taskDefinition']['taskDefinitionArn']
            
            print(f"   ✓ Task definition registered")
            print(f"     ARN: {task_def_arn}")
            
            # Store for later
            self.task_definition_arn = task_def_arn
        
        except Exception as e:
            print(f"   ✗ Task definition registration failed: {e}")
            raise
    
    def run_task(self):
        """Run ECS task"""
        print("\n[OPTIONAL] Running ECS task...")
        
        cluster_name = f"{self.project_name}-cluster-{self.environment}"
        
        try:
            # Get subnet and security group from CloudFormation outputs
            stack_name = f"{self.project_name}-{self.environment}"
            response = self.cloudformation.describe_stacks(StackName=stack_name)
            outputs = {o['OutputKey']: o['OutputValue'] for o in response['Stacks'][0]['Outputs']}
            
            subnet_id = outputs['PrivateSubnet1Id']
            sg_id = outputs['SecurityGroupId']
            
            # Run task
            response = self.ecs.run_task(
                cluster=cluster_name,
                taskDefinition=self.task_definition_arn,
                launchType='FARGATE',
                networkConfiguration={
                    'awsvpcConfiguration': {
                        'subnets': [subnet_id],
                        'securityGroups': [sg_id],
                        'assignPublicIp': 'DISABLED'
                    }
                }
            )
            
            task_arn = response['tasks'][0]['taskArn']
            print(f"   ✓ Task started: {task_arn}")
        
        except Exception as e:
            print(f"   ✗ Task execution failed: {e}")
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Deploy GATE AE Pipeline to AWS'
    )
    
    parser.add_argument(
        '--environment',
        choices=['development', 'staging', 'production'],
        default='production',
        help='Deployment environment'
    )
    
    parser.add_argument(
        '--region',
        default='ap-south-1',
        help='AWS region'
    )
    
    parser.add_argument(
        '--profile',
        help='AWS profile to use'
    )
    
    parser.add_argument(
        '--ecr-only',
        action='store_true',
        help='Only create ECR repository and push image'
    )
    
    parser.add_argument(
        '--cloudformation-only',
        action='store_true',
        help='Only deploy CloudFormation stack'
    )
    
    parser.add_argument(
        '--alarms-only',
        action='store_true',
        help='Only setup CloudWatch alarms'
    )
    
    parser.add_argument(
        '--run-task',
        action='store_true',
        help='Run ECS task after deployment'
    )
    
    args = parser.parse_args()
    
    # Create deployer
    deployer = AWSDeployer(
        environment=args.environment,
        region=args.region,
        profile=args.profile
    )
    
    try:
        # Execute based on flags
        if args.ecr_only:
            deployer.create_ecr_repository()
            deployer.build_and_push_image()
        
        elif args.cloudformation_only:
            deployer.deploy_cloudformation()
        
        elif args.alarms_only:
            deployer.setup_cloudwatch_alarms()
        
        else:
            # Full deployment
            deployer.deploy_all()
        
        # Optionally run task
        if args.run_task:
            deployer.run_task()
        
        print("\n✅ Deployment successful!")
        return 0
    
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())