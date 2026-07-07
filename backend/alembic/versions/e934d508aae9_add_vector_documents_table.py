"""add vector_documents table

Revision ID: e934d508aae9
Revises: ffebf6ec97b8
Create Date: 2026-07-07 22:16:18.212381

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import pgvector.sqlalchemy
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'e934d508aae9'
down_revision: Union[str, None] = 'ffebf6ec97b8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # NOTE: autogenerate also proposed dropping checkpoint_blobs/checkpoint_writes/
    # checkpoint_migrations/checkpoints — those are owned and created by
    # langgraph-checkpoint-postgres's own AsyncPostgresSaver.setup(), not by our
    # SQLModel metadata, so they were stripped from this migration. Alembic will
    # keep proposing to drop them on every future autogenerate; that's expected
    # and those lines should always be removed, never applied.
    op.create_table('vector_documents',
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('collection', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('content', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('doc_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('embedding', pgvector.sqlalchemy.Vector(dim=1536), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_vector_documents_collection'), 'vector_documents', ['collection'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_vector_documents_collection'), table_name='vector_documents')
    op.drop_table('vector_documents')
