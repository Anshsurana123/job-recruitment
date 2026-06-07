package com.example.salesvoice.ui.products

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.example.salesvoice.R
import com.example.salesvoice.data.model.Product
import com.example.salesvoice.databinding.ItemProductBinding
import com.google.android.material.chip.Chip

class ProductAdapter(
    private val currencySymbol: String,
    private val onEditClick: (Product) -> Unit,
    private val onDeleteClick: (Product) -> Unit
) : ListAdapter<Product, ProductAdapter.ProductViewHolder>(ProductDiffCallback()) {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ProductViewHolder {
        val binding = ItemProductBinding.inflate(
            LayoutInflater.from(parent.context), parent, false
        )
        return ProductViewHolder(binding)
    }

    override fun onBindViewHolder(holder: ProductViewHolder, position: Int) {
        holder.bind(getItem(position))
    }

    inner class ProductViewHolder(private val binding: ItemProductBinding) :
        RecyclerView.ViewHolder(binding.root) {

        fun bind(product: Product) {
            val context = binding.root.context
            binding.tvProductName.text = product.name
            binding.tvProductPrice.text = "${product.unit} — $currencySymbol${product.pricePerUnit}/${product.unit}"
            binding.tvProductProfit.text = context.getString(R.string.sale_profit_format, "$currencySymbol${product.profitPerUnit}/${product.unit}")

            binding.cgAliases.removeAllViews()
            
            product.getAliasList().forEach { alias ->
                val chip = Chip(context).apply {
                    text = alias
                    isCheckable = false
                    isClickable = false
                    setEnsureMinTouchTargetSize(false)
                    chipMinHeight = context.resources.getDimension(R.dimen.alias_chip_height)
                    setTextAppearance(R.style.AliasChipTextAppearance)
                    chipBackgroundColor = context.getColorStateList(R.color.accent_grey_light)
                    chipStrokeWidth = 0f
                }
                binding.cgAliases.addView(chip)
            }

            binding.btnEdit.setOnClickListener { onEditClick(product) }
            binding.btnDelete.setOnClickListener { onDeleteClick(product) }
        }
    }

    class ProductDiffCallback : DiffUtil.ItemCallback<Product>() {
        override fun areItemsTheSame(oldItem: Product, newItem: Product): Boolean {
            return oldItem.id == newItem.id
        }

        override fun areContentsTheSame(oldItem: Product, newItem: Product): Boolean {
            return oldItem == newItem
        }
    }
}
