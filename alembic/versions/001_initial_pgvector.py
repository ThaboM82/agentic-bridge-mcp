"""initial_pgvector_setup

Revision ID: 001_initial
Revises: 
Create Date: 2026-10-28 09:34:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Enable the pgvector extension in PostgreSQL
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 2. Create workflow_executions table
    op.create_table(
        'workflow_executions',
        sa.Column('id', PG_UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column('execution_id', sa.String(length=64), nullable=False),
        sa.Column('brief_id', sa.String(length=64), nullable=False),
        sa.Column('current_step', sa.String(length=64), nullable=False, default='initialized'),
        sa.Column('status', sa.String(length=32), nullable=False, default='running'),
        sa.Column('extracted_requirements', sa.JSON(), nullable=True),
        sa.Column('architectural_specification', sa.JSON(), nullable=True),
        sa.Column('design_tokens', sa.JSON(), nullable=True),
        sa.Column('execution_logs', sa.JSON(), nullable=True),
        sa.Column('current_error', sa.Text(), nullable=True),
        sa.Column('brief_embedding', Vector(1536), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False)
    )

    # 3. Create indexes for high-performance querying
    op.create_index('ix_workflow_executions_id', 'workflow_executions', ['id'], unique=True)
    op.create_index('ix_workflow_executions_execution_id', 'workflow_executions', ['execution_id'], unique=True)
    op.create_index('ix_workflow_executions_brief_id', 'workflow_executions', ['brief_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_workflow_executions_brief_id', table_name='workflow_executions')
    op.drop_index('ix_workflow_executions_execution_id', table_name='workflow_executions')
    op.drop_index('ix_workflow_executions_id', table_name='workflow_executions')
    op.drop_table('workflow_executions')
    op.execute("DROP EXTENSION IF EXISTS vector;")
