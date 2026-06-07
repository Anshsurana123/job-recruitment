package com.example.salesvoice.ui.record

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.example.salesvoice.R
import com.example.salesvoice.data.model.SaleEntry
import com.example.salesvoice.databinding.ItemSaleLogBinding
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class SaleLogAdapter(
    private val currencySymbol: String,
    private val onDeleteClick: (SaleEntry) -> Unit
) : ListAdapter<SaleEntry, SaleLogAdapter.SaleViewHolder>(SaleDiffCallback()) {

    private val timeFormat = SimpleDateFormat("hh:mm a", Locale.getDefault())

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): SaleViewHolder {
        val binding = ItemSaleLogBinding.inflate(
            LayoutInflater.from(parent.context), parent, false
        )
        return SaleViewHolder(binding)
    }

    override fun onBindViewHolder(holder: SaleViewHolder, position: Int) {
        holder.bind(getItem(position))
    }

    inner class SaleViewHolder(private val binding: ItemSaleLogBinding) :
        RecyclerView.ViewHolder(binding.root) {

        fun bind(sale: SaleEntry) {
            val date = Date(sale.timestamp)
            binding.tvSaleTime.text = timeFormat.format(date)
            binding.tvSaleProductName.text = sale.productName
            binding.tvSaleQuantity.text = "${sale.quantity} ${sale.unit}"
            binding.tvSaleAmount.text = "$currencySymbol${String.format(Locale.getDefault(), "%.2f", sale.totalAmount)}"
            binding.tvSaleProfit.text = binding.root.context.getString(R.string.sale_profit_format, "$currencySymbol${String.format(Locale.getDefault(), "%.2f", sale.totalProfit)}")

            binding.btnDeleteSale.setOnClickListener { onDeleteClick(sale) }
        }
    }

    class SaleDiffCallback : DiffUtil.ItemCallback<SaleEntry>() {
        override fun areItemsTheSame(oldItem: SaleEntry, newItem: SaleEntry): Boolean {
            return oldItem.id == newItem.id
        }

        override fun areContentsTheSame(oldItem: SaleEntry, newItem: SaleEntry): Boolean {
            return oldItem == newItem
        }
    }
}
