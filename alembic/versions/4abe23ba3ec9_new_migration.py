"""New Migration

Revision ID: 4abe23ba3ec9
Revises: 
Create Date: 2022-05-19 23:59:37.305540

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '4abe23ba3ec9'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('Order',
    sa.Column('order_id', sa.Integer(), nullable=False),
    sa.Column('request', sa.String(), nullable=True),
    sa.Column('time_created', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('time_updated', sa.DateTime(timezone=True), nullable=True),
    sa.PrimaryKeyConstraint('order_id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('Order')
    # ### end Alembic commands ###
