"""Alustava migraatio tietokannan luomiseksi

Revision ID: abc123456789
Revises: 
Create Date: 2025-05-14 01:20:00
"""

from alembic import op
import sqlalchemy as sa

# Migraation tunniste
revision = 'abc123456789'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Luo taulukot tietokantaan
    op.create_table('data_entries',
        sa.Column('id', sa.Text, primary_key=True),
        sa.Column('data_type', sa.Text, nullable=False),
        sa.Column('content', sa.Text, nullable=False),
        sa.Column('source', sa.Text, nullable=False),
        sa.Column('validated', sa.Integer, nullable=False),
        sa.Column('metadata', sa.Text, nullable=False),
        sa.Column('quality_score', sa.Float, nullable=False)
    )
    op.create_table('embeddings',
        sa.Column('sentence_id', sa.Text, primary_key=True),
        sa.Column('entry_id', sa.Text, nullable=False),
        sa.Column('sentence', sa.Text, nullable=False),
        sa.Column('embedding', sa.LargeBinary, nullable=False),
        sa.ForeignKeyConstraint(['entry_id'], ['data_entries.id'])
    )
    op.create_table('history',
        sa.Column('id', sa.Text, primary_key=True),
        sa.Column('entry_id', sa.Text, nullable=False),
        sa.Column('content', sa.Text, nullable=False),
        sa.Column('timestamp', sa.Text, nullable=False),
        sa.ForeignKeyConstraint(['entry_id'], ['data_entries.id'])
    )

def downgrade():
    # Poista taulukot tietokannasta
    op.drop_table('history')
    op.drop_table('embeddings')
    op.drop_table('data_entries')