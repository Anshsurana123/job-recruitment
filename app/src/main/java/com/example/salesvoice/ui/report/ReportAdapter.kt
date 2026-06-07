package com.example.salesvoice.ui.report

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.example.salesvoice.databinding.ItemReportRowBinding
import java.util.Locale

class ReportAdapter(
    private val currencySymbol: String
) : ListAdapter<ProductBreakdown, ReportAdapter.ReportViewHolder>(ReportDiffCallback()) {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ReportViewHolder {
        val binding = ItemReportRowBinding.inflate(
            LayoutInflater.from(parent.context), parent, false
        )
        return ReportViewHolder(binding)
    }

    override fun onBindViewHolder(holder: ReportViewHolder, position: Int) {
        holder.bind(getItem(position))
    }

    inner class ReportViewHolder(private val binding: ItemReportRowBinding) :
        RecyclerView.ViewHolder(binding.root) {

        fun bind(breakdown: ProductBreakdown) {
            binding.tvReportProductName.text = breakdown.productName
            binding.tvReportQuantity.text = "${breakdown.totalQuantity} ${breakdown.unit}"
            binding.tvReportRevenue.text = "$currencySymbol${String.format(Locale.getDefault(), "%.2f", breakdown.totalRevenue)}"
            binding.tvReportProfit.text = "$currencySymbol${String.format(Locale.getDefault(), "%.2f", breakdown.totalProfit)}"
        }
    }

    class ReportDiffCallback : DiffUtil.ItemCallback<ProductBreakdown>() {
        override fun areItemsTheSame(oldItem: ProductBreakdown, newItem: ProductBreakdown): Boolean {
            return oldItem.productName == newItem.productName
        }

        override fun areContentsTheSame(oldItem: ProductBreakdown, newItem: ProductBreakdown): Boolean {
            return oldItem == newItem
        }
    }
}
